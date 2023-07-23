from flask import *  
from video2frames import *

# initialize the flask app
app = Flask(__name__, static_url_path="/static") 


# define the / route that render the index.html
@app.route('/')  
def main():  
    return render_template("index.html") 


@app.route('/test')  
def test():  
    return render_template("test.html") 

@app.route('/process', methods = ['POST'])  
def success():  
    if request.method != 'POST':
        return # return the 405 http response
    
    # get the uploaded file
    file = request.files['file']
    file_path = "dataset" + file.filename
    # save the file to defined path
    file.save(file_path)
    # convert the video into frames
    convert(file_path)
    return render_template("Acknowledgement.html", name = file.filename)  
  
if __name__ == '__main__':  
    app.run(debug=True)