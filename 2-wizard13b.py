from onprem import LLM
import sys

if __name__ == '__main__':
    llm = LLM(use_larger=True, n_gpu_layers=49)

    if (len(sys.argv) == 1):
        llm.prompt("What is VSHN AppOps?")
    else:
        llm.ingest("./sample_data")
        question = """What is VSHN AppOps? Remember to only use the provided context."""
        result = llm.ask(question)
        #print(result)
        print("\nSources:\n")
        for i, document in enumerate(result["source_documents"]):
            print(f"\n{i+1}.> " + document.metadata["source"] + ":")
            print(document.page_content)
