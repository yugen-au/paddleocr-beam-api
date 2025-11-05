from beam import PythonVersion, Image, Sandbox

# Quick test to verify Beam is working
sandbox = Sandbox(
    name="quickstart", 
    image=Image(python_version=PythonVersion.Python311)
)

sb = sandbox.create()

result = sb.process.run_code("print('hello from the sandbox!')").result

print(result)

sb.terminate()
