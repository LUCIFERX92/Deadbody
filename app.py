# app.py

import gradio as gr
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

# Create the Gradio Interface
iface = gr.Interface(
    fn=generate_responses, 
    inputs=gr.inputs.Textbox(lines=10, placeholder="Enter the email content here..."), 
    outputs=["text", "text", "text"]
)

# Launch the Interface
iface.launch()
