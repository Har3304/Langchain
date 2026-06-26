import logging
import json
import time
from datetime import datetime, timezone
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from google.colab import userdata
import os

# --- Configuration & Logging Setup ---
token = userdata.get('HF_TOKEN')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = token
os.environ['HF_TOKEN'] = token

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName
        }
        if hasattr(record, 'extra_data'):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)
  
def setup_logging():
    logger = logging.getLogger('langgraph_app')
    logger.setLevel(logging.INFO)
    # Clear existing handlers to prevent duplicate logs in Colab notebooks
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    return logger

# --- Metrics Collector Class ---
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'requests_total': 0,
            'errors_total': 0,
            'latency_sum': 0,
            'latency_count': 0,
            'tokens_input': 0,
            'tokens_output': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    # FIX: Indentation corrected to sit inside the class
    def record_request(self, latency_ms: float, input_tokens: int, output_tokens: int, error: bool = False, cache_hit: bool = False):
        self.metrics['requests_total'] += 1
        self.metrics['latency_sum'] += latency_ms
        self.metrics['latency_count'] += 1
        self.metrics['tokens_input'] += input_tokens
        self.metrics['tokens_output'] += output_tokens
        if error:
            self.metrics['errors_total'] += 1
        if cache_hit:
            self.metrics['cache_hits'] += 1  # FIX: Removed the buggy duplicate increment line
        else:
            self.metrics['cache_misses'] += 1
      
    # FIX: Indentation corrected
    def get_summary(self):
        avg_latency = (self.metrics['latency_sum'] / self.metrics['latency_count']) if self.metrics['latency_count'] > 0 else 0
        error_rate = (self.metrics['errors_total'] / self.metrics['requests_total']) if self.metrics['requests_total'] > 0 else 0
        cache_hit_rate = (self.metrics['cache_hits'] / self.metrics['requests_total']) if self.metrics['requests_total'] > 0 else 0
        return {
            'total_requests': self.metrics['requests_total'],
            'total_errors': self.metrics['errors_total'],
            'average_latency_ms': round(avg_latency, 2),
            'total_tokens_input': self.metrics['tokens_input'],
            'total_tokens_output': self.metrics['tokens_output'],
            'cache_hit_rate': round(cache_hit_rate, 2),
            'error_rate': round(error_rate, 2)
        }

# --- Instrumented LLM Class ---
class InstrumentedLLM:
    def __init__(self):
        # Initializing the LLM endpoint wrapper
        self.llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B', timeout=30))
        self.metrics = MetricsCollector()
        self.logger = setup_logging()

    def invoke(self, query: str, **kwargs):
        start_time = time.time()  # FIX: Changed from datetime.now to work with Unix math calculation
        
        try:
            # Simple text length token estimations
            input_tokens = len(query.split()) * 4 // 3
            
            response = self.llm.invoke(query)
            result = response.content
            output_tokens = len(result.split()) * 4 // 3
            
            latency = (time.time() - start_time) * 1000
            
            self.metrics.record_request(
                latency_ms=latency, 
                input_tokens=input_tokens, 
                output_tokens=output_tokens,
                error=False,
                cache_hit=False
            )
            
            self.logger.info(
                'LLM request completed',
                extra={'extra_data': {
                    'latency_ms': round(latency, 2),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                }}
            )
            return result

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            # FIX: Call correct method name `record_request` instead of `record`
            self.metrics.record_request(
                latency_ms=latency,
                input_tokens=0,
                output_tokens=0,
                error=True,
                cache_hit=False
            )
            # FIX: Pulled inside the except block scope so it correctly logs during a failure
            self.logger.error(
                'LLM request failed',
                extra={'extra_data': {
                    'error': str(e),
                    'latency_ms': round(latency, 2)
                }}
            )
            raise

# --- The Demo Function ---
def demo_monitoring():
    print("--- Initializing Instrumented LLM Application ---")
    instrumented_model = InstrumentedLLM()
    
    # 1. Test standard successful queries
    sample_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one simple sentence.",
        "Write a 3-word slogan for an AI logging tool."
    ]
    
    print("\n--- Simulating Live Successful Production Traffic ---")
    for idx, prompt in enumerate(sample_prompts, 1):
        print(f"\n[Running Query {idx}/3]: '{prompt}'")
        try:
            response = instrumented_model.invoke(prompt)
            print(f"Response: {response}")
        except Exception:
            print("Query execution failed.")
        time.sleep(1) # Gap requests slightly
        
    # 2. Simulate a failure scenario (e.g. invalid arguments or empty runtime requests)
    print("\n--- Simulating an Application Error Scenario ---")
    try:
        # Forcing a model invocation layer error by passing an invalid empty execution block
        instrumented_model.invoke(query="")
    except Exception as error:
        print(f"[Captured Expected Exception Domain]: Outbound call rejected or failed.")

    # 3. Print out final collected system metric telemetry
    print("\n" + "="*50)
    print("FINAL TELEMETRY METRICS SUMMARY REPORT")
    print("="*50)
    summary = instrumented_model.metrics.get_summary()
    print(json.dumps(summary, indent=4))
    print("="*50)

# Run the system monitoring showcase
if __name__ == '__main__':
    demo_monitoring()
