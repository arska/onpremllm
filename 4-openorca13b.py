from onprem import LLM
import sys

if __name__ == '__main__':
    url = "https://huggingface.co/kroonen/OpenOrca-Platypus2-13B-GGUF/resolve/main/OpenOrca-Platypus2-13B-Q4_K_M.gguf"
    llm = LLM(url, n_gpu_layers=49)

    template = """
### Instruction:

{prompt}

### Response:"""

    if (len(sys.argv) == 1):
        answer = llm.prompt(
            "What is VSHN AppOps?", prompt_template=template
        )
    else:
        answer = llm.ask("What is VSHN AppOps?")
        print("\nSources:\n")
        for i, document in enumerate(answer["source_documents"]):
            print(f"\n{i+1}.> " + document.metadata["source"] + ":")
            print(document.page_content)
