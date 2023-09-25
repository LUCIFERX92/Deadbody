from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Initialize the text generation pipeline with trust_remote_code=True
text_gen_pipeline = pipeline("text-generation", model="h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-3b", trust_remote_code=True)

# Generate Responses Function
def generate_responses(email_content):
    # Generate three different responses
    response_1 = text_gen_pipeline(email_content, max_length=100, do_sample=True, top_k=50)[0]['generated_text']
    response_2 = text_gen_pipeline(email_content, max_length=100, do_sample=True, top_k=50)[0]['generated_text']
    response_3 = text_gen_pipeline(email_content, max_length=100, do_sample=True, top_k=50)[0]['generated_text']
    
    return response_1, response_2, response_3

# Input the email content
email_content = input("Enter the email content: ")

# Get the responses
responses = generate_responses(email_content)

# Print the responses
print("\nGenerated Responses:")
for i, response in enumerate(responses, 1):
    print(f"\nResponse {i}: {response}")
