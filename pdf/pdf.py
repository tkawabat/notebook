from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import (
    LAParams,
    LTContainer,
    LTTextLine,
)

def get_objs(layout, results):
    if not isinstance(layout, LTContainer):
        return
    for obj in layout:
        if isinstance(obj, LTTextLine):
            results.append({'bbox': obj.bbox, 'text' : obj.get_text(), 'type' : type(obj)})
        get_objs(obj, results)

def main(path):
    with open(path, "rb") as f:
        parser = PDFParser(f)
        document = PDFDocument(parser)
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed
        # https://pdfminersix.readthedocs.io/en/latest/api/composable.html#
        laparams = LAParams(
            all_texts=True,
        )
        rsrcmgr = PDFResourceManager()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        page_num = 0
        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            layout = device.get_result()
            results = []
            print('objs-------------------------')
            get_objs(layout, results)
            for r in results:
                print(r["bbox"], r["text"])
            
            page_num += 1
            if page_num > 5:
                break



main('pdf/230ぬれぎぬ.pdf')