def _build_vs(fname, chunk_size=0, persist_directory=None, 
            max_pages=0, verbose=False):
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(fname)
    pages = loader.load()

    print(f'len(pages) = {len(pages)}')

    if chunk_size > 0:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        docs = r_splitter.split_documents(pages)
    else:
        docs = pages

    print(f'len(docs) = {len(docs)}')

    if max_pages > 0:
        docs = docs[:max_pages]
        print(f'len(docs) for vs = {len(docs)}')

    from langchain.embeddings import HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings()

    from langchain.vectorstores import Chroma

    # NOTE: Chroma.from_documents auto combines all docs, must delete the db first
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )
    vectordb.delete_collection()

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=persist_directory
    )
    if persist_directory is not None:
        vectordb.persist()
    print('vectordb count {}'.format(vectordb._collection.count()))
    print(f'vectordb for {fname} done!')

    return vectordb

def _get_hash(str_or_file, is_file=False):
    import hashlib
    if not is_file:
        return hashlib.md5(str_or_file.encode('utf8')).hexdigest()
    else:
        with open(str_or_file, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
        return file_hash.hexdigest()


def _build_vs_dedup(fname, chunk_size=0, persist_directory=None, collection_name=None,
            max_pages=0, verbose=False):
    """Each file per collection and deduplicated."""

    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(fname)
    pages = loader.load()

    print(f'len(pages) = {len(pages)}')

    if chunk_size > 0:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        docs = r_splitter.split_documents(pages)
    else:
        docs = pages

    from collections import defaultdict
    counter = defaultdict(int)

    for doc in docs:
        mdata = doc.metadata
        key = '{}:{}'.format(mdata['source'], mdata['page'])
        mdata['block'] = counter[key]
        # mdata['source'] = '{}|page:{}|block:{}'.format(mdata['source'], mdata['page'], counter[key])
        # mdata['source'] = 'p{}_b{}_{}'.format(mdata['page'], counter[key], mdata['source'])
        counter[key] += 1

    # import pdb; pdb.set_trace()

    print(f'len(docs) = {len(docs)}')

    if max_pages > 0:
        docs = docs[:max_pages]
        print(f'len(docs) for vs = {len(docs)}')

    from langchain.embeddings import HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings()

    from langchain.vectorstores import Chroma

    if collection_name is None:
        collection_name = _get_hash(fname, is_file=True)
    ids = [_get_hash(p.page_content) for p in docs]

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_directory,
        # 1. by default it is L2, 2. cosine is distance here, cosine_similarity = 1 - cosine_distance
        collection_metadata={"hnsw:space": "cosine"} 
    )
    print(f"vector db {vectordb._collection.name} has {vectordb._collection.count()} records: {fname}")
    existing_ids = set(vectordb.get()['ids'])
    docs_dedup = {_id: doc for _id, doc in zip(ids, docs) if _id not in existing_ids}

    if len(docs_dedup)  > 0:
        vectordb = Chroma.from_documents(
            documents=list(docs_dedup.values()),
            ids=list(docs_dedup.keys()),
            collection_name=collection_name,
            embedding=embedding,
            persist_directory=persist_directory,
        )
    if persist_directory is not None:
        vectordb.persist()
    print(f"updated vector db {vectordb._collection.name} has {vectordb._collection.count()} records: {fname}")
    return vectordb

def _load_vs(persist_directory=None):
    from langchain.embeddings import HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings()

    from langchain.vectorstores import Chroma
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )
    return vectordb

def _build_vs_collection(folder, collection_name, chunk_size=0, persist_directory=None,
            max_pages=0, verbose=False):
    import os, glob
    DEFAULT_COLLECTION = folder
    vectordb = None
    if os.path.exists(DEFAULT_COLLECTION):
        pdfs = sorted(glob.glob(os.path.join(DEFAULT_COLLECTION, '**', '*.pdf'), recursive=True))
        for pdf in pdfs:
            vectordb = _build_vs_dedup(pdf, collection_name=collection_name,
                    chunk_size=chunk_size, persist_directory=persist_directory, max_pages=max_pages, verbose=verbose)
    return vectordb

PROMPT_TEMPLATE_QA = """"Use the following pieces of context and chat history to answer the question.

{context}"""

def test_qa_similarity_search():
    from llms import _llm_call
    plain_message = 'summarize the text'

    # build vector store
    vectordb = _build_vs_dedup('test_files/flash_attention_v2.pdf')
    # similarity search
    res = vectordb.similarity_search(plain_message, k=3)

    # import pdb; pdb.set_trace()

    # stuff context and llm call for response
    context = '\n\n'.join([doc.page_content for doc in res])
    system_prompt = PROMPT_TEMPLATE_QA.format(context=context)
    _kwargs = {'system_prompt': system_prompt} # overwrite system_prompt
    bot_message = _llm_call(plain_message, [], **_kwargs)
    print(bot_message)

def test_qa_similarity_search_collection():
    from llms import _llm_call
    plain_message = 'summarize the text'

    # # build vector store
    # vectordb = _build_vs_dedup('test_files/flash_attention_v2.pdf')
    vectordb = _build_vs_collection('data/default_collection', collection_name='default_collection')

    # similarity search
    res = vectordb.similarity_search_with_score(plain_message, k=3)
    scores = [1.0 - _r[1] for _r in res] # extract scores
    res = [_r[0] for _r in res]

    # stuff context and llm call for response
    context = '\n\n'.join([doc.page_content for doc in res])
    system_prompt = PROMPT_TEMPLATE_QA.format(context=context)
    _kwargs = {'system_prompt': system_prompt} # overwrite system_prompt
    # setup llama-2-7b
    from dotenv import load_dotenv
    load_dotenv()
    import llms
    llms.parse_endpoints_from_environ()
    # for storenet files, set chat_engine to llama-2
    bot_message = _llm_call(plain_message, [], chat_engine='llama-2-7b-chat-hf', **_kwargs)
    bot_message += "## Sources\n{}".format([doc.metadata for doc in res])
    bot_message += '\n{}'.format(scores)
    print(bot_message)


def test_qa_retrievalqa():
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA

    llm = ChatOpenAI()

    # build vector store
    vectordb = _build_vs_dedup('test_files/flash_attention_v2.pdf')

    retriever = vectordb.as_retriever()
    qa_stuff = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        verbose=True
    )
    query =  "summerize the text"
    response = qa_stuff.run(query)
    print(response)

def test_qa_retrievalqawithsources():
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQAWithSourcesChain
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain

    llm = ChatOpenAI()

    # build vector store
    vectordb = _build_vs_dedup('test_files/flash_attention_v2.pdf', chunk_size=2048) # NOTE: can not fill with single PDF page, too large

    qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff",)
    retriever = vectordb.as_retriever()
    qa = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=retriever)
    query = 'why flash attention v2 is better than v1?'
    response = qa({"question": query})
    print(response)


if __name__ == '__main__':
    # test_qa_similarity_search()
    test_qa_similarity_search_collection()
    # test_qa_retrievalqa()
    # test_qa_retrievalqawithsources()
    
