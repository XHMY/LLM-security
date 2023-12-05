from llama_cpp import Llama


class LlamaModel():
    def __init__(self, model_name="llama-2-70b",
                 add_system_prompt=True,
                 model_path="/nfs/stak/users/zengyif/hpc-share/opt/llama.cpp/models/llama-2-70b-chat.Q5_K_M.gguf") -> None:
        self.model_name = model_name
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=100,
            n_ctx=1024)
        self.add_system_prompt = add_system_prompt

    def fit_message(self, msg):
        if self.add_system_prompt:
            conversation = [
                "You are a helpful assistant.",
                f"User: {msg}",
                "Assistant: "
            ]
        else:
            conversation = [
                f"User: {msg}",
                "Assistant: "
            ]
        return "\n".join(conversation)

    def __call__(self, msg, **kwargs):
        while True:
            try:
                raw_response = self.llm(
                    self.fit_message(msg),
                    stop=['User:'],
                    **kwargs)
                self.raw_response = raw_response
                return raw_response['choices'][0]['text']
            except Exception as e:
                print(e)
                pass


if __name__ == '__main__':
    model = LlamaModel(
        model_name="llama-2-7b",
        add_system_prompt=True,
        model_path="/Volumes/MacGM7/Models/llama-2-7b-chat.Q5_K_M.gguf")
    msg = "I am a good person."
    print(model(msg))
    print(model.raw_response)
