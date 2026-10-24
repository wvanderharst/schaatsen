"""Zet web_model.json in de template en schrijft web/langebaan.html."""
import os
h = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(h, 'web', 'langebaan_template.html'), encoding='utf-8').read()
d = open(os.path.join(h, 'web_model.json'), encoding='utf-8').read().replace('</', '<\\/')
open(os.path.join(h, 'web', 'langebaan.html'), 'w', encoding='utf-8').write(t.replace('/*__DATA__*/null', d))
print('web/langebaan.html geschreven')
