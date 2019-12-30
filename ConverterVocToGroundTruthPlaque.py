#-*- coding: utf-8 -*-

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict
import time
import os


class XMLHandler:
    def __init__(self, xml_path: str or Path):
        self.xml_path = Path(xml_path)
        self.root = self.__open()

    def __open(self):
        with self.xml_path.open() as opened_xml_file:
            self.tree = ET.parse(opened_xml_file)
            return self.tree.getroot()

def CategorieToClasseId(Categorie):
    if (Categorie == "palette"): return 0

def converter(xml_files: str, output_folder: str) -> None:
    xml_files = sorted(list(Path(xml_files).rglob("*.xml")))
    #On ouvre le fichier en lecture
    fichier = open("output.manifest.temp","w+")
    for xml_index, xml in enumerate(xml_files, start=1):
        xml_content = XMLHandler(xml)
        for index,sg_box in enumerate(xml_content.root.iter('annotation')):
            fichier.write("{\"source-ref\":\"" + sg_box.find("path").text + "\",")
            fichier.write("\"" + sg_box.find("source").find("database").text + "\":{")
            fichier.write("\"annotations\":[")
            for index, sg_box_ in enumerate(xml_content.root.iter('object')):
                #On calcule les boudind boxes
                width = int(sg_box_.find("bndbox").find("xmax").text) - int(sg_box_.find("bndbox").find("xmin").text)
                height = int(sg_box_.find("bndbox").find("ymax").text) - int(sg_box_.find("bndbox").find("ymin").text)
                top = int(sg_box_.find("bndbox").find("ymin").text)
                left = int(sg_box_.find("bndbox").find("xmin").text)

                fichier.write("{\"class_id\":" + str(CategorieToClasseId(sg_box_.find("name").text)) + ",")
                fichier.write("\"width\":" + str(width) + ",")
                fichier.write("\"top\":" + str(top) + ",")
                fichier.write("\"height\":" + str(height) + ",")
                fichier.write("\"left\":" + str(left) + "},")
            fichier.write("],")

            fichier.write("\"image_size\":[{")
            fichier.write("\"width\":" + sg_box.find("size").find("width").text + ",\"depth\":" + sg_box.find("size").find("depth").text + ",\"height\":" + sg_box.find("size").find("height").text)
            fichier.write("}]},\"" + sg_box.find("source").find("database").text + "-metadata\":{\"job-name\":\"labeling-job/sortPalette\",")
            fichier.write("\"class-map\":{\"0\":\"palette\"},")
            fichier.write("\"human-annotated\":\"yes\",\"type\":\"groundtruth/object-detection\"}")
            fichier.write("}")
            fichier.writelines("\n")
    fichier.close()

    #On remplace ,] par ]
    fin = open('output.manifest.temp', 'r+')
    fout = open('output.manifest', 'w+')

    for line in fin:
	    fout.write(line.replace(',]', ']'))
    fin.close()
    fout.close()
    os.remove('output.manifest.temp')

    
if __name__ == '__main__':
    t1 = time.time()
    XML_FOLDER = "DataSet"
    OUTPUT_FOLDER =  "."

    converter(XML_FOLDER, OUTPUT_FOLDER)
    print('Temps de Traitement : %d ms'%((time.time()-t1)*1000))
