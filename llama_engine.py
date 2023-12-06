from llama_cpp import Llama


class LlamaModel:
    def __init__(self,
                 system_prompt="You are a helpful assistant.",
                 model_path="/nfs/stak/users/zengyif/hpc-share/opt/llama.cpp/models/llama-2-70b-chat.Q5_K_M.gguf") -> None:
        self.system_prompt = system_prompt
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=100,
            n_ctx=1024)

    def fit_message(self, msg):
        if self.system_prompt is not None:
            conversation = [
                self.system_prompt,
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
    model = LlamaModel(model_path="/Volumes/MacGM7/Models/llama-2-7b-chat.Q5_K_M.gguf")
    msg = "I am a good person."
    print(model(msg))
    print(model.raw_response)
