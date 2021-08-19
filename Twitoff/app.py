from flask import Flask, render_template
from flask import request
from .models import db, User, Tweet
from .twitter import insert_example_users, add_or_update_user, prediction_model, vectorize_tweet
import os


# creates application
def create_app():

    app = Flask(__name__, template_folder='templates/')
    app.config['DEBUG'] = True
    app.config['EXPLAIN_TEMPLATE_LOADING'] = True

    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    # Create tables
    with app.app_context():
        db.create_all()

    @app.route('/', methods=["POST","GET"])
    def home():
        name = request.form.get('user_name')
        if name:
            add_or_update_user(name)
            print(name)
        
        return render_template('home.html')

    @app.route('/about')
    def about():
       return render_template('about.html')

    @app.route('/reset')
    def reset():
        db.drop_all()
        db.create_all()
        return render_template('home.html')

    @app.route('/iris')
    def iris():    
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        X, y = load_iris(return_X_y=True)
        clf = LogisticRegression(random_state=0, solver='lbfgs',
                            multi_class='multinomial').fit(X, y)

        return str(clf.predict(X[:2, :]))

    @app.route('/guess', methods=["POST","GET"])
    def guess():
        winner = ''
        name1 = request.form.get('user1')
        name2 = request.form.get('user2')
        fake_tweet = request.form.get('fake_tweet')
        if name1 and name2 and fake_tweet:
            result = prediction_model(name1, name2, fake_tweet)
            if result == 0.0:
                winner = name1.upper()
            elif result == 1.0:
                winner = name2.upper()
        
        return render_template('guess.html', prediction=winner)



    return app