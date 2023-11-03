from onprem import LLM
import sys

if __name__ == '__main__':
    url = "https://huggingface.co/hadongz/falcon-7b-instruct-gguf/resolve/main/falcon-7b-instruct-q4_0.gguf"

    llm = LLM(url, n_gpu_layers=49)

    template = """### Instruction:

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
