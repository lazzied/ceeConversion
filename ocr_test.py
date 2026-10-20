from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    device="cpu"
)

output = pipeline.predict(
    "./data/input/exam.pdf"
)

for res in output:
    res.save_to_markdown("./data/output/paddleocr")     