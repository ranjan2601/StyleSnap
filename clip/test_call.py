import requests

# Upload an image
with open("hd.jpeg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/tag",
        files={"file": ("test.jpg", f, "image/jpeg")}
    )
    print(response.json())
    with open("response.json", "w") as out_file:
        out_file.write(response.text)