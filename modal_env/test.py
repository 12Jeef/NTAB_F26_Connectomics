import modal

app = modal.App("getting-started")

@app.function()
def square(x: int) -> int:
  print("Using cloud compute for advanced mathematics!")
  return x**2

@app.local_entrypoint()
def main(x: int = 42):
  print(f"The square of {x} is {square.remote(x)}")
