from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

import numpy as np
import plotly.graph_objects as go

import tensorflow as tf

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

def local_mins(x):
    return 1.2*(x + 0.05)*(x - 0.05) - 2*(x - .2)**2*(x + .2)**2 + (x - .3)**3*(x + .3)**3

functions = {
    'bowl': lambda x, y: x**2 + y**2,
    'double_bowl': lambda x, y: (x-0.25)**2*(x+0.25)**2 - (x-0.25)*(x+0.25) + y**2,
    'sin_cos': lambda x, y: tf.cos(np.pi*x)*tf.sin(np.pi*y),
    'local_mins': lambda x, y: local_mins(tf.sqrt(tf.square(x) + tf.square(y)))
}

optimizers = {
    'SGD': tf.keras.optimizers.SGD,
    'RMSprop': tf.keras.optimizers.RMSprop,
    'Adam': tf.keras.optimizers.Adam,
    'Adadelta': tf.keras.optimizers.Adadelta,
    'Adagrad': tf.keras.optimizers.Adagrad,
    'Adamax': tf.keras.optimizers.Adamax,
    'Nadam': tf.keras.optimizers.Nadam,
    'Ftrl': tf.keras.optimizers.Ftrl
}

app = Dash(__name__)
server = app.server

background_color = ''

app.layout = html.Div(
    children=[
        html.H1(
            'Visualizing Learning Optimizers',
            style={
                'textAlign': 'center',
                'padding': '30px'
            }
        ),
        html.Br(),
        dcc.Markdown(
            'Visualize how the Keras optimizers traverse a learning surface. Source code can be found on [Github](https://github.com/KyroChi/keras-optimizers-visualization).',
            style={'textAlign': 'center'}
        ),
        dcc.Markdown(
            'Written by [K. R. Chickering](https://www.math.ucdavis.edu/~krc/).',
            style={
                'textAlign': 'center',
            }
        ),
        html.Br(),
        html.Table(
            children=[
                html.Tr([
                    html.Td([
                        html.Label('Optimizer'),
                        dcc.Dropdown(
                            list(optimizers.keys()),
                            list(optimizers.keys())[0],
                            multi=False,
                            clearable=False,
                            id='optimizer-dropdown'
                        ),
                        html.Br(),
                        html.Label('Learning Surface'),
                        dcc.Dropdown(
                            list(functions.keys()),
                            list(functions.keys())[0],
                            multi=False,
                            clearable=False,
                            id='learning-surface-dropdown'
                        ),
                        html.Br(),
                        html.Label('Learning Rate'),
                        dcc.Input(
                            value=0.01,
                            type='number',
                            min=1e-05,
                            max=100,
                            id='learning-rate'
                        ),
                        html.Br(),
                        html.Label('Starting x-coord'),
                        dcc.Input(
                            value=0.25,
                            type='number',
                            id='start-x'
                        ),
                        html.Br(),
                        html.Label('Starting y-coord'),
                        dcc.Input(
                            value=-0.95,
                            type='number',
                            id='start-y'
                        ),
                        html.Br(),
                        html.Label('Number of Steps'),
                        dcc.Input(
                            value=30,
                            type='number',
                            min=1,
                            max=1000,
                            id='num-steps'
                        ),
                        html.Br(),
                        html.Label(
                            'Momentum',
                            id='mom-label'
                        ),
                        dcc.RadioItems(
                            ['True', 'False'],
                            'False',
                            inline=True,
                            id='mom'
                        ),
                        html.Br(),
                        html.Label(
                            'Nesterov',
                            id='nesterov-label'
                        ),
                        dcc.RadioItems(
                            ['True', 'False'],
                            'False',
                            inline=True,
                            id='nesterov'
                        ),
                        html.Br(),
                        html.Button(
                            'Optimize',
                            id='go-button'
                        )
                    ],
                            style={
                                'width': '35%',
                                'textAlign': 'left',
                                'padding': '30px'
                            }
                    ),
                    html.Td([
                        dcc.Graph(
                            id='learning-surface',
                            style={
                                'height': '80vh',
                            }
                        )
                    ],
                            style={
                                'textAlign': 'center',
                                'height': '60%'
                            }
                    )
                ])
            ],
            style={
                'textAlign': 'center',
                'width': '100%'
            }
        )
    ])

@app.callback(
    Output('nesterov', 'style'),
    Output('mom', 'style'),
    Output('nesterov-label', 'style'),
    Output('mom-label', 'style'),
    Input('optimizer-dropdown', 'value')
)
def show_nesterov_and_mom(optimizer):
    if optimizer == 'SGD':
        
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


@app.callback(
    Output('learning-surface', 'figure'),
    State('optimizer-dropdown', 'value'),
    State('learning-surface-dropdown', 'value'),
    State('learning-rate', 'value'),
    State('start-x', 'value'),
    State('start-y', 'value'),
    State('num-steps', 'value'),
    State('mom', 'value'),
    State('nesterov', 'value'),
    Input('go-button', 'n_clicks')
)
def update_graph(optimizer, surface, lr, start_x, start_y,
                 num_steps, mom, nesterov,  n_clicks):
    mom = True if mom == 'True' else False
    nesterov = True if nesterov == 'True' else False
    
    var_x = tf.Variable(start_x)
    var_y = tf.Variable(start_y)
    loss = lambda: functions[surface](var_x, var_y)
    
    # TODO: Must parse the optimizer so I don't pass bad args.
    if optimizer == 'SGD':
        opt = optimizers[optimizer](
            learning_rate=lr,
            momentum=mom,
            nesterov=nesterov
        )
    else:
        opt = optimizers[optimizer](
            learning_rate=lr,
        )

    x_steps = [var_x.value().numpy()]
    y_steps = [var_y.value().numpy()]

    for step in range(num_steps):
        step_count = opt.minimize(loss, [var_x, var_y]).numpy()
        x_steps.append(var_x.value())
        y_steps.append(var_y.value())
    
    Z = functions[surface](X, Y)
    z_pts = [functions[surface](x_steps[ii], y_steps[ii]) for ii, _ in enumerate(x_steps)]
    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale='Gray',
                opacity=0.45,
                colorbar=None
            ),
            go.Scatter3d(
                x=x_steps,
                y=y_steps,
                z=z_pts,
                mode='markers',
                marker=dict(
                    size=3,
                    color=[ii for ii, _ in enumerate(x_steps)],
                    colorscale='Viridis',
                    colorbar_x=1.15
                )
            )
        ],
        layout=go.Layout(
            paper_bgcolor='rgba(0,0,0,0)'
        )
    )
    fig.update_yaxes(range=[-np.abs(start_x), np.abs(start_x)]),
    fig.update_xaxes(range=[-np.abs(start_y), np.abs(start_y)])
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
