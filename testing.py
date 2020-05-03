import timeit

import numpy as np

#cy = timeit.timeit('example_cy.test(5)', setup = 'import example_cy', number = 100)
#py = timeit.timeit('example_py.test(5)', setup = 'import example_py', number = 100)

cy = timeit.timeit('detect36C.detect(np.array(misc.imresize(Image.open("testPhoto.jpg"), [480,640])))', setup = 'import detect36C; import numpy as np; from PIL import Image; from scipy import misc', number = 1)
#py = timeit.timeit('detect36.detect(np.random.rand(240,320,3))', setup = 'import detect36; import numpy as np', number = 1)

print(cy)
#print('Cython is {}x faster'.format(py/cy))
