import secrets

from flask import Flask, jsonify, request, render_template, url_for

from nlu.nlu_engine import Brain
from nlu.tokens import Tokens

app = Flask(__name__)
  
tk = Tokens()
brain = Brain()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/<api_key>/<query>')
def api(api_key, query):
    infered_list = []
    """
    token = request.args.get('token')
    query = request.args.get('query')
    verified = tk.verify_token(token)"""

    infered_list = brain.chat(query)
    return {'token': api_key,
                    'query': query,
                    'response': infered_list}



@app.route('/generate_token')
def gen():
    address = request.remote_addr
    print('address: '+ address)
    _token = tk.verify_token(address)

    if not _token:
        __token = secrets.token_urlsafe(16)
        tk.data_entry(address, __token)
        return render_template('token.html', token=__token)
    else:
        return render_template('token.html', token=_token)

        """
        with open('tokens', 'rb') as tokens:
            token_dict = pickle.load(tokens)
        if address in token_dict.keys():
            return render_template('token.html', token=token_dict[address])
        else:
            token_dict[address] = secrets.token_urlsafe(16)
            pickle.dump('tokens', token_dict)
            return render_template('token.html', token=token_dict[address])

    else:
        token_dict = {}
        token_dict[address] = secrets.token_urlsafe(16)
        with open('tokens', 'wb') as tokens:
            pickle.dump(token_dict, tokens, protocol=pickle.HIGHEST_PROTOCOL)
            """

@app.route('/docs')
def docs():
    return render_template('docs.html')

