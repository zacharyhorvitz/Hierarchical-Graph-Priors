# import pandas as pd
# import re
# import spacy
from openie import StanfordOpenIE

def process_text(text,name):

    with StanfordOpenIE() as client:
        print('Text: %s.' % text)
        for triple in client.annotate(text):
            print('|-', triple)

        graph_image = name+'_graph.png'
        client.generate_graphviz_graph(text, graph_image)
        print('Graph generated: %s.' % graph_image)

        # with open('corpus/pg6130.txt', 'r', encoding='utf8') as r:
        #     corpus = r.read().replace('\n', ' ').replace('\r', '')

        # triples_corpus = client.annotate(corpus[0:50000])
        # print('Corpus: %s [...].' % corpus[0:80])
        # print('Found %s triples in the corpus.' % len(triples_corpus))
        # for triple in triples_corpus[:3]:
        #     print('|-', triple)


