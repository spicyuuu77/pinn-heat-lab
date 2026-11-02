from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

base = Path(__file__).resolve().parent
ns = {'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
with ZipFile(base/'candidate_results.pptx') as source, ZipFile(base/'candidate_results_fixed.pptx','w',ZIP_DEFLATED) as out:
    for item in source.infolist():
        payload = source.read(item.filename)
        if '/charts/' in item.filename and item.filename.endswith('.xml'):
            root = E.fromstring(payload)
            if root.find('.//c:scatterChart',ns) is not None:
                series = root.findall('.//c:scatterChart/c:ser',ns)
                display = '0.0' if len(series) > 1 else '0'
                for axis in root.findall('.//c:valAx',ns):
                    fmt = axis.find('c:numFmt',ns)
                    if fmt is None:
                        raise ValueError('Missing explicit axis number format')
                    fmt.set('formatCode', display)
                    fmt.set('sourceLinked', '0')
                payload = E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
        out.writestr(item,payload)
print('Axis display formats fixed; original numeric data preserved.')
