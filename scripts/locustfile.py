from locust import HttpUser, task, between, events
import random
import json
import os

# Load sample feature vector for testing
SAMPLE = None
SAMPLE_PATH = "features/sample_feature.json"

def init_sample():
    """Load or create sample feature vector."""
    global SAMPLE
    if os.path.exists(SAMPLE_PATH):
        try:
            with open(SAMPLE_PATH, "r") as f:
                SAMPLE = json.load(f)
            print(f"[+] Loaded sample from {SAMPLE_PATH}")
        except Exception as e:
            print(f"[!] Error loading sample: {e}")
            SAMPLE = [0.0] * 22  # Default 22 features
    else:
        # Default: 22 features of reasonable scale
        SAMPLE = [0.0] * 22
        print(f"[*] Using default sample with {len(SAMPLE)} features")


def make_sample(anom_rate=0.05):
    """Generate a sample, optionally with anomaly perturbation."""
    if SAMPLE is None:
        init_sample()
    
    x = SAMPLE.copy() if isinstance(SAMPLE, list) else list(SAMPLE)
    
    # Create anomaly by perturbing random features
    if random.random() < anom_rate:
        num_to_perturb = random.randint(1, max(1, len(x) // 4))
        for _ in range(num_to_perturb):
            idx = random.randint(0, len(x) - 1)
            # Large perturbation to trigger reconstruction error
            x[idx] = x[idx] * 50.0 + random.gauss(0, 100)
    
    return x


# Initialize sample on module load
init_sample()


class IDSUser(HttpUser):
    """Locust user class for load testing IDS service."""
    
    wait_time = between(0.01, 0.2)  # Random wait 10-200ms between requests
    
    @task(weight=8)
    def score_single(self):
        """Single sample scoring (80% of traffic)."""
        payload = {"features": make_sample(anom_rate=0.05)}
        with self.client.post(
            "/score", 
            json=payload, 
            catch_response=True
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")
    
    @task(weight=2)
    def score_batch(self):
        """Batch scoring (20% of traffic)."""
        batch_size = random.randint(10, 100)
        samples = [make_sample(anom_rate=0.05) for _ in range(batch_size)]
        payload = {"features": samples}
        with self.client.post(
            "/batch_score", 
            json=payload, 
            catch_response=True
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")
    
    @task(weight=1)
    def check_health(self):
        """Health check (10% of traffic)."""
        with self.client.get(
            "/health",
            catch_response=True
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("\n" + "="*70)
    print("IDS Service Load Test Started")
    print(f"Sample shape: {len(SAMPLE)} features" if SAMPLE else "Sample shape: unknown")
    print("="*70 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("\n" + "="*70)
    print("IDS Service Load Test Completed")
    print("="*70 + "\n")
