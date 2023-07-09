from website import create_app
from generator1 import generateImage1
from generator2 import generateImage2
from mobilegenerator import generateImageM
from firstGenerator import generateFirst
app = create_app([generateFirst,generateImage1,generateImage2,generateImageM])

if __name__ == '__main__':
    app.run(debug=True)