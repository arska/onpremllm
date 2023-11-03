from onprem import LLM
import sys

if __name__ == '__main__':
    url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
    llm = LLM(url, n_gpu_layers=49)
    template = """
[INST] {prompt} [/INST]
"""
    
    if (len(sys.argv) == 1):
        answer = llm.prompt(
            "What is VSHN AppOps?", prompt_template=template
        )    
    else:
        llm.ingest("./sample_data")
        answer = llm.ask("What is VSHN AppOps?")
        print("\nSources:\n")
        for i, document in enumerate(answer["source_documents"]):
            print(f"\n{i+1}.> " + document.metadata["source"] + ":")
            print(document.page_content)
