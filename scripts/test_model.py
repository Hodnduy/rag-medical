import time
from mlx_lm import load, generate

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

def main():
    print(f"[*] Load model {MODEL_ID}")
    start_load = time.time()
    
    model, tokenizer = load(MODEL_ID)
    
    print(f"[+] Loading complete: {time.time() - start_load:.2f} s.")

    messages = [
        {"role": "system", "content": "You are a professor in computer science."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("\n[*] Generate result")
    start_gen = time.time()
    
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=300,
        verbose=True
    )
    
    total_time = time.time() - start_gen
    print(f"\n[+] Total time: {total_time:.2f} second.")

if __name__ == "__main__":
    main()