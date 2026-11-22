with open("official.py", "r") as f:
    content = f.read()
lines = content.split("\n")
libraries = []
for line in lines:
    if not line.startswith("get_ipython().run_line_magic('pip', 'install "):
        continue
    libs = line[len("get_ipython().run_line_magic('pip', 'install ") : -2].split(" ")
    libs = [lib for lib in libs if not lib.startswith("-")]
    libraries.extend(libs)
lines = [line for line in lines if not line.startswith("get_ipython")]
lines = [line.replace("display(", "print(") for line in lines]
with open("official.py", "w") as f:
    f.write("\n".join(lines))

print(libraries)
