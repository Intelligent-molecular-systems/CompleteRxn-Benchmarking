"""Minimal benchmark workflow. Run with: python examples/quickstart.py"""
from pathlib import Path
import tempfile

from completerxn import DataLoader


class MyModel:
    def predict(self, inputs):
        # Replace this with your model: return one completed reaction per input.
        return list(inputs)


if __name__ == '__main__':
    # Load the published HF data and splits; files are cached after download.
    train, val, test = DataLoader().load(split='random', rep=0)
    test = test[:10]

    predictions = test.predict(MyModel())
    result = test.evaluate(predictions)
    print(result.summary)
    print(result.failure_summary())

    # Save scores, predictions, failure examples, and reaction drawings.
    output = Path(tempfile.mkdtemp(prefix='completerxn-example-'))
    result.save(output / 'results')
    result.plot(test[0].reaction_id).save(output / 'reaction.svg')
    result.plot_failures().save(output / 'failures.svg')
    print(f'Saved outputs to {output}')
