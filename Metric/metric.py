class COCOCaptionEvaluator:
    def __init__(self):
        import language_evaluation
        self.evaluator = language_evaluation.CocoEvaluator(verbose=False)


    def evaluate(self, predicts, answers):

        results = self.evaluator.run_evaluation(predicts, answers)

        return results

predict = ['The cat sat on the mat']
target = ['The is cat on the mat']

evaluator = COCOCaptionEvaluator()
print(evaluator.evaluate(predict,target))