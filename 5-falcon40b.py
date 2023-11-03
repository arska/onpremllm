from onprem import LLM
import sys

if __name__ == '__main__':
    # url = "https://huggingface.co/maddes8cht/tiiuae-falcon-40b-instruct-gguf/resolve/main/ggml-tiiuae-falcon-40b-instruct-Q3_K_S.gguf"
    url = "https://huggingface.co/YokaiKoibito/falcon-40b-GGUF/resolve/main/falcon-40b-Q3_K_M.gguf"
    llm = LLM(url, n_gpu_layers=49)

    template = """
### Instruction:

{prompt}

### Response:"""

    if (len(sys.argv) == 1):
        answer = llm.prompt(
            "As a software developer, what is VSHN AppOps?"
        )
    else:
        answer = llm.ask("As a software developer, what is VSHN AppOps?", prompt_template=template)
        print("\nSources:\n")
        for i, document in enumerate(answer["source_documents"]):
            print(f"\n{i+1}.> " + document.metadata["source"] + ":")
            print(document.page_content)
