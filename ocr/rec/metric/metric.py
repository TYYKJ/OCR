import Levenshtein


class RecMetric:
    def __init__(self, converter):
        """
        文本识别相关指标计算类

        :param converter: 用于label转换的转换器
        """
        self.converter = converter

    def __call__(self, predictions, labels):
        n_correct = 0
        norm_edit_dis = 0.0
        predictions = predictions.softmax(dim=2).detach().cpu().numpy()
        preds_str = self.converter.decode(predictions)
        # show_str = []
        # pred_str = []
        # label_str = []
        for (pred, pred_conf), target in zip(preds_str, labels):
            norm_edit_dis += Levenshtein.distance(pred, target) / max(len(pred), len(target))
            # pred_str.append(f'{pred}')
            # label_str.append(f'{label_str}')
            if pred == target:
                n_correct += 1

        return {'n_correct': n_correct, 'norm_edit_dis': norm_edit_dis}
        # return {'n_correct': n_correct, 'norm_edit_dis': norm_edit_dis, 'show_str':
        #         pd.DataFrame({'pred': pred_str[:5], 'label': label_str[:5]})}
