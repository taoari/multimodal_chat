import inspect
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
                    upload_btn = gr.Button("📁", scale=0.1, min_width=0, interactive=True)
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

        # NOTE: gr.State will not show in API
        fake_api_btn.click(self._api_fn, [textbox, chatbot_state], [fake_response, chatbot_state], api_name='chat')

        self._setup_submit(textbox.submit, textbox, chatbot)
        self._setup_submit(submit_btn.click, textbox, chatbot)

        retry_btn.click(self._retry, [chatbot], [chatbot])
        undo_btn.click(self._undo, [textbox, chatbot], [textbox, chatbot]) # NOTE: state can not undo or retry
        clear_btn.click(self._clear, None, [chatbot, chatbot_state])

    def _setup_submit(self, event_trigger, textbox, chatbot):
        # https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
        chat_msg = event_trigger(self._add_message, [textbox, chatbot], [textbox, chatbot], api_name=False)
        if self.is_generator:
            bot_msg = chat_msg.then(self._bot_stream_fn, chatbot, chatbot, api_name='chat_with_history')
        else:
            bot_msg = chat_msg.then(self._bot_fn, chatbot, chatbot, api_name='chat_with_history')
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