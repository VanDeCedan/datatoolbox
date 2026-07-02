import sys
import app
import tempfile

class MockFile:
    def __init__(self, name):
        self.name = name

def test_ocr():
    f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    f.close()
    
    # We pass a mock file object because gradio passes a wrapper or path
    # app._get_path will use the 'name' attribute or path
    res = list(app.run_ocr_mode([MockFile(f.name)], 50))
    print(res[-1])

if __name__ == "__main__":
    test_ocr()
