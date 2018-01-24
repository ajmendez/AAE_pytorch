from sacred import Experiment

ex = Experiment('hello_config')


@ex.config
def my_config():
    recipient = 'world'
    message = 'Hello {}!'.format(recipient)


@ex.automain
def my_main(message):
    print(message)

