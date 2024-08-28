import inspect
import re
import gradio as gr

def _convert_history(history, type='tuples'):
    assert type in ['tuples', 'messages']
    if len(history) > 0:
        history_type = 'messages' if isinstance(history[0], dict) else 'tuples'
        if history_type == type:
            return history
        
        if history_type == 'messages':
            res = []
            for msg in history:
                if msg['role'] == 'user':
                    res.append((msg, None))
                else:
                    res.append((None, msg))
        else:
            for user, bot in history:
                if user is not None:
                    res.append({'role': 'user', 'content': user})
                if bot is not None:
                    res.append({'role': 'assistant', 'content': bot})
    return res
    
class ChatInterface(gr.Blocks):
    def __init__(self, 
                fn,
                *,
                additional_inputs=[],
                additional_outputs=[],
                type='tuples',
                multimodal=False,
                avatar_images=(None, None),
                analytics_enabled=None,
                css=None,
                title=None,
                theme=None,):
        super().__init__(
            analytics_enabled=analytics_enabled,
            mode="chat_interface",
            css=css,
            title=title or "Gradio",
            theme=theme,
        )
        chatbot = gr.Chatbot(type=type, avatar_images=avatar_images)
        chatbot_state = gr.State([])

        with gr.Group():
            with gr.Row():
                if not multimodal:
                    upload_btn = gr.UploadButton("📁", scale=0.1, min_width=0, interactive=True)
                    audio_btn = gr.Button("🎤", scale=0.1, min_width=0, interactive=True)
                    textbox = gr.Textbox(placeholder="Type a message...", scale=7, show_label=False, container=False, interactive=True)
                    submit_btn = gr.Button("Submit", variant="primary", scale=1, min_width=0, interactive=True)
                    fake_response = gr.Textbox(visible=False)
                else:
                    audio_btn = gr.Button("🎤", scale=0.1, min_width=0, interactive=True)
                    textbox = gr.MultimodalTextbox(placeholder="Type a message...", file_count="multiple",
                            scale=10, show_label=False, container=False, interactive=True)
                    fake_response = gr.MultimodalTextbox(visible=False)

        with gr.Row():
            retry_btn = gr.Button("Retry", scale=1, min_width=0)
            undo_btn = gr.Button("Undo", scale=1, min_width=0)
            clear_btn = gr.Button("Clear", scale=1, min_width=0)

        fake_api_btn = gr.Button("Fake API", visible=False)
        stop_btn = gr.Button("Stop", visible=False, variant="stop", scale=1, min_width=0, interactive=True)

        self.type = type
        self.fn = fn
        self.is_generator = inspect.isgeneratorfunction(self.fn) 
        
        self.chatbot = chatbot
        self.chatbot_state = chatbot_state
        self.upload_btn = upload_btn
        self.audio_btn = audio_btn
        self.textbox = textbox
        self.submit_btn = submit_btn
        self.retry_btn = retry_btn
        self.undo_btn = undo_btn
        self.clear_btn = clear_btn

        # NOTE: gr.State will not show in API
        fake_api_btn.click(self._api_fn, [textbox, chatbot_state], [fake_response, chatbot_state], api_name='chat')

        self._setup_submit(textbox.submit, textbox, chatbot, api_name='chat_with_history')
        self._setup_submit(submit_btn.click, textbox, chatbot, api_name=False)

        retry_btn.click(self._retry, [chatbot], [chatbot], api_name=False)
        undo_btn.click(self._undo, [textbox, chatbot], [textbox, chatbot], api_name=False) # NOTE: state can not undo or retry
        clear_btn.click(self._clear, None, [chatbot, chatbot_state], api_name=False)

        upload_btn.upload(
            self._upload_fn, 
            [textbox, upload_btn], 
            [textbox], queue=False, api_name='upload')


    def _setup_submit(self, event_trigger, textbox, chatbot, api_name='chat_with_history'):
        # https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
        chat_msg = event_trigger(self._add_message, [textbox, chatbot], [textbox, chatbot], api_name=False)
        if self.is_generator:
            bot_msg = chat_msg.then(self._bot_stream_fn, chatbot, chatbot, api_name=api_name)
        else:
            bot_msg = chat_msg.then(self._bot_fn, chatbot, chatbot, api_name=api_name)
        bot_msg.then(lambda: gr.update(interactive=True), None, [textbox], api_name=False)

    def _add_message(self, message, history):
        # message: str (Textbox) or dict (gr.MultimodalTextbox)
        # history: tuples or messages
        message = {'text': message, 'files': []} if isinstance(message, str) else message
        for f in message["files"]:
            history.append({'role': 'user', 'content': (f,)})
        if message["text"] is not None:
            history.append({'role': 'user', 'content': message["text"]})
        return gr.update(value=None, interactive=False), history
    
    def _bot_fn(self, history, *args):
        message = history[-1]['content']
        response = self.fn(message, history[:-1], *args)
        history.append({'role': 'assistant', 'content': response})
        return history
    
    def _bot_stream_fn(self, history, *args):
        message = history[-1]['content']
        response = self.fn(message, history[:-1], *args)
        history.append({'role': 'assistant', 'content': None})
        for _response in response:
            history[-1]['content'] = _response
            yield history

    def _api_fn(self, message, chat_state, *args):
        if self.is_generator:
            *_, response = self.fn(message, chat_state, *args)
        else:
            response = self.fn(message, chat_state, *args)
        return response, chat_state + [{'role': 'user', 'content': message}, {'role': 'assistant', 'content': response}]
    
    def _retry(self, history):
        if len(history) >= 2:
            message = history[-2]['content']
            _history = history[:-2]
            if self.is_generator:
                # clear history first
                yield history[:-1]

                response = self.fn(message, _history)
                for _response in response:
                    history[-1]['content'] = _response
                    yield history
            else:
                history[-1]['content'] = self.fn(message, _history)
                yield history
        else:
            yield history
                
    def _undo(self, textbox, history):
        if len(history) >= 2:
            message = history[-2]['content']
            return message, history[:-2]
        return textbox, history

    def _clear(self):
        return gr.update(value=[]), gr.update(value=[])
    
    def _upload_fn(self, msg, filepath_or_filename):
        if filepath_or_filename is None:
            return msg
        filename = filepath_or_filename.name if hasattr(filepath_or_filename, 'name') else filepath_or_filename
        import os, mimetypes
        from utils.message import render_message
        mtype = mimetypes.guess_type(filename)[0]
        if mtype.startswith('image'):
            msg_dict = {'images': [filename]}
        elif mtype.startswith('audio'):
            msg_dict = {'audios': [filename]}
        elif mtype.startswith('video'):
            msg_dict = {'videos': [filename]}
        else:
            msg_dict = {'files': [filename]}
        msg_dict['text'] = msg
        return msg + '\n' + re.sub(r'^\s+', '', render_message(msg_dict)).replace('\n', '')


def reload_javascript():
    """reload custom javascript. The following code enables bootstrap css and makes chatbot message buttons responsive.
    """
    print("Reloading javascript...")
    js = """
<!-- for bootstrap -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-9ndCyUaIbzAi2FUVXJi0CjmCapSmO7SnpJef0486qhLnuZ2cdeRhO02iuK6FUUVM" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-geWF76RCwLtnZ8qwWowPQNguL3RmwHVBC9FhGdlKrxdiJJigb/j/68SIy3Te4Bkz" crossorigin="anonymous"></script>

<!-- for message buttons to work -->
<script>
function registerMessageButtons() {
	const collection = document.querySelectorAll(".btn-chatbot");
	for (let i = 0; i < collection.length; i++) {
      // NOTE: gradio use .value instead of .innerHTML for gr.Textbox
	  collection[i].onclick=function() {
        elem = document.getElementById("inputTextBox").getElementsByTagName('textarea')[0];
        elem.value = collection[i].getAttribute("value"); // use value instead of innerHTML
        elem.dispatchEvent(new Event('input', {
            view: window,
            bubbles: true,
            cancelable: true
            }))
        };  
	}
}
// need to make sure registerMessageButtons() is executed all the time as new message can come out;
var intervalId = window.setInterval(function(){
  registerMessageButtons();
}, 1000);
</script>
"""
    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        # soup = BeautifulSoup(res.body, 'html.parser')
        # # NOTE: gradio UI is rendered by JavaScript, so can not find btn-chatbot
        # res.body = str(soup).replace('</html>', f'{js}</html>').encode('utf8')
        res.body = res.body.replace(b'</html>', f'{js}</html>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response

GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse