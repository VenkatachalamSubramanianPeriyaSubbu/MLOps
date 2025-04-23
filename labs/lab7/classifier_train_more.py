from metaflow import FlowSpec, step, kubernetes

class ClassifierTrainFlow(FlowSpec):

    @kubernetes(cpu=0.5, memory=512)  # add for cloud execution
    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        import numpy as np

        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X, y, test_size=0.2, random_state=0)
        print("Data loaded successfully")
        self.lambdas = np.arange(0.001, 1, 0.2)
        self.next(self.train_lasso, foreach='lambdas')

    @kubernetes(cpu=0.5, memory=512)
    @step
    def train_lasso(self):
        from sklearn.linear_model import Lasso

        self.model = Lasso(alpha=self.input)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @kubernetes(cpu=0.5, memory=512)
    @step
    def choose_model(self, inputs):
        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.next(self.end)

    @kubernetes(cpu=0.5, memory=512)
    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)

if __name__ == '__main__':
    ClassifierTrainFlow()
