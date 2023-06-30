# Multimodal Chatbot
Multimodal Chatbot based on Gradio Template.

## Install

* Create `.env` under the project root folder with `OPENAI_API_KEY`

```
OPENAI_API_KEY=<openai_api_key>
```

* You need to install the zbar library

```
apt install zbar-tools # Linux
brew install zbar # Mac OS X
```

* Create a folder named `data/`, and copy `walgreens_full_import_passthru.csv` to `data/`

* Setup a conda environment and run

```
conda create -n gradio python=3.10
conda activate gradio

pip install -r requirement.txt
python app.py
```

