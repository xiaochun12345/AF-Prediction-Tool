"""Data-only inference for the locked revision_v2 Elastic Net ensemble."""
import json
import math
from pathlib import Path


class InputError(ValueError):
    pass


def load_model(path=None):
    path = Path(path) if path else Path(__file__).with_name('model.json')
    model = json.loads(path.read_text(encoding='utf-8'))
    if model['format_version'] != 2 or len(model['fits']) != 5:
        raise ValueError('Expected the five-model revision_v2 export.')
    return model


def validate_inputs(model, values):
    cleaned, errors = {}, []
    for field in model['features']:
        key = field['name']
        try:
            value = float(values[key])
        except (KeyError, TypeError, ValueError):
            errors.append(field['label'])
            continue
        if not math.isfinite(value):
            errors.append(field['label'])
        elif field['type'] == 'category' and value not in field['choices']:
            errors.append(field['label'] + ' (invalid category)')
        elif field['type'] == 'number' and field.get('minimum') is not None and value < field['minimum']:
            errors.append(field['label'] + ' (below allowed minimum)')
        else:
            cleaned[key] = value
    if errors:
        raise InputError('Complete or correct: ' + '; '.join(errors))
    return cleaned


def fit_risk(fit, values):
    lp = -fit['offset'] + math.fsum(
        term['coefficient'] * (values[term['feature']] if term['category'] is None
                               else float(values[term['feature']] == term['category']))
        for term in fit['terms'])
    result = []
    for hazard in fit['baseline_hazard']:
        if hazard == 0:
            result.append(0.)
        else:
            log_hazard = math.log(hazard) + lp
            result.append(1. if log_hazard > 700 else -math.expm1(-math.exp(log_hazard)))
    return result


def predict(model, values):
    values = validate_inputs(model, values)
    each = [fit_risk(fit, values) for fit in model['fits']]
    return [math.fsum(column) / len(each) for column in zip(*each)]


def outside_training_range(model, values):
    return [field['label'] for field in model['features'] if field['type'] == 'number'
            and not field['training_min'] <= values[field['name']] <= field['training_max']]


def _score_inputs(values, numeric, binary):
    out = {}
    for name in numeric + binary:
        try:
            value = float(values[name])
        except (KeyError, TypeError, ValueError):
            raise InputError('Complete all score inputs before calculating.') from None
        if not math.isfinite(value) or value < 0 or (name in binary and value not in [0., 1.]):
            raise InputError('Invalid score input: ' + name)
        out[name] = value
    return out


def charge_af_5y(values):
    x = _score_inputs(values, ['age','height','weight','sbp','dbp'],
                      ['race','current_smoking','antihypertensive_use','diabete','heartfailer','infarc'])
    lp = (.5083*x['age']/5 + .46491*x['race'] + .2478*x['height']*100/10
          + .1155*x['weight']/15 + .1972*x['sbp']/20 - .1013*x['dbp']/10
          + .35931*x['current_smoking'] + .34889*x['antihypertensive_use']
          + .23666*x['diabete'] + .70127*x['heartfailer'] + .49659*x['infarc'])
    exponent = lp - 12.5815600
    return 1. if exponent > 700 else -math.expm1(math.log(.9718412736)*math.exp(exponent))


def harms2_af(values):
    x = _score_inputs(values, ['age','bmi','alcohol_units_week'],
                      ['hypertension','sex','sleepapnoea','ever_smoking'])
    return int(4*x['hypertension'] + (2 if x['age'] >= 65 else 1 if x['age'] >= 60 else 0)
               + (x['bmi'] >= 30) + 2*x['sex'] + 2*x['sleepapnoea'] + x['ever_smoking']
               + (2 if x['alcohol_units_week'] >= 15 else 1 if x['alcohol_units_week'] >= 7 else 0))
