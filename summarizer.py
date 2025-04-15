from transformers import pipeline

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    input_text = input("Enter the text you want to summarize:\n")
    print("\nSummary:")
    print(summarize_text(input_text))
