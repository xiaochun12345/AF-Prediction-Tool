"""Verified labels and coding for the revision_v2 website export."""
NUMERIC = {
 'age': ('Age','years','Demographics & body measures'),
 'bmi': ('Body mass index','kg/m²','Demographics & body measures'),
 'sbp': ('Systolic blood pressure','mmHg','Demographics & body measures'),
 'weight': ('Weight','kg','Demographics & body measures'),
 'height': ('Height','m','Demographics & body measures'),
 'numoftreatments_medications': ('Number of baseline treatments / medications','count','Medical history'),
 'duration': ('Time since recorded COPD report','years','COPD & medication use'),
 'Albumin': ('Albumin','g/L','Biomarkers & AF-PRS'),
 'Alkaline_phosphatase': ('Alkaline phosphatase','U/L','Biomarkers & AF-PRS'),
 'Apolipoprotein_B': ('Apolipoprotein B','g/L','Biomarkers & AF-PRS'),
 'C_reactive_protein': ('C-reactive protein','mg/L','Biomarkers & AF-PRS'),
 'Cholesterol': ('Total cholesterol','mmol/L','Biomarkers & AF-PRS'),
 'Creatinine': ('Creatinine','µmol/L','Biomarkers & AF-PRS'),
 'Cystatin_C': ('Cystatin C','mg/L','Biomarkers & AF-PRS'),
 'Gamma_glutamyltransferase': ('Gamma-glutamyltransferase','U/L','Biomarkers & AF-PRS'),
 'Glucose': ('Glucose','mmol/L','Biomarkers & AF-PRS'),
 'Glycated_haemoglobin': ('HbA1c','mmol/mol','Biomarkers & AF-PRS'),
 'LDL_direct': ('LDL cholesterol (direct)','mmol/L','Biomarkers & AF-PRS'),
 'Urate': ('Urate','µmol/L','Biomarkers & AF-PRS'),
 'Urea': ('Urea','mmol/L','Biomarkers & AF-PRS'),
 'prs': ('AF polygenic risk score','same AF-PRS scale as the study','Biomarkers & AF-PRS'),
}
BINARY = {
 'sleepapnoea': ('Sleep apnoea','Lifestyle & symptoms'),
 'lossweight': ('Weight loss versus one year ago','Lifestyle & symptoms'),
 'disability': ('Long-standing illness, disability or infirmity','Medical history'),
 'Blood_clot_in_the_leg': ('History of deep vein thrombosis','Medical history'),
 'Hayfever_allergic_rhinitis_eczema': ('Hay fever, allergic rhinitis or eczema','Medical history'),
 'Angina': ('Angina','Medical history'),
 'hypertension': ('Hypertension','Medical history'),
 'diabete': ('Diabetes','Medical history'),
 'heartfailer': ('Heart failure','Medical history'),
 'stroke': ('Stroke','Medical history'),
 'infarc': ('Myocardial infarction','Medical history'),
 'j44_prior': ('Inpatient J44.1 record on or before baseline','COPD & medication use'),
 'SABA': ('Short-acting beta agonist (SABA)','COPD & medication use'),
 'LABA': ('Long-acting beta agonist (LABA)','COPD & medication use'),
 'SAMA': ('Short-acting muscarinic antagonist (SAMA)','COPD & medication use'),
 'LAMA': ('Long-acting muscarinic antagonist (LAMA)','COPD & medication use'),
 'Theophylline': ('Theophylline','COPD & medication use'),
 'Systemic_corticosteroids': ('Systemic corticosteroids','COPD & medication use'),
 'Beta_blockers': ('Beta blockers','COPD & medication use'),
 'NonDHP_CCB': ('Non-dihydropyridine calcium channel blocker','COPD & medication use'),
 'ACE_inhibitors': ('ACE inhibitors','COPD & medication use'),
 'ARBs': ('Angiotensin receptor blockers (ARBs)','COPD & medication use'),
 'Digoxin': ('Digoxin','COPD & medication use'),
}
SPECIAL = {
 'sex': ('Sex','Demographics & body measures', {0:'Female',1:'Male'}),
 'race': ('Ethnic background','Demographics & body measures', {0:'Other / non-white',1:'White'}),
 'smoke': ('Smoking history','Lifestyle & symptoms', {0:'Never',1:'Current or previous'}),
 'fall': ('Falls in the last year','Lifestyle & symptoms', {0:'No falls',1:'One fall',2:'More than one fall'}),
 'insomnia': ('Trouble falling asleep or waking at night','Lifestyle & symptoms',
              {0:'Usually',1:'Never / rarely or sometimes'}),
}
HELP = {
 'duration':'Baseline assessment date minus UKB field 42016, divided by 365.25. This is recorded duration, not a verified biological onset date.',
 'j44_prior':'First recorded inpatient J44.1 diagnosis on or before baseline. This is not the number of exacerbations or a complete one-year exacerbation history.',
 'prs':'Use the same AF-PRS definition and scale as the study. A score from another genetic test cannot be substituted directly.',
 'insomnia':'Study coding is 0 = usually, 1 = never/rarely or sometimes; the displayed choices map to the verified study values.',
 'Glycated_haemoglobin':'Enter mmol/mol, not percent. The previous website unit label was incorrect.',
 'Urate':'Enter µmol/L, not mmol/L.',
}


def metadata(name):
    if name in NUMERIC:
        label, unit, group = NUMERIC[name]
        return {'name':name,'label':label,'unit':unit,'group':group,'type':'number',
                'minimum':None if name=='prs' else 0.,'help':HELP.get(name,'Use the baseline measurement.')}
    if name in BINARY:
        label, group = BINARY[name]
        choices={0:'No',1:'Yes'}
    else:
        label, group, choices = SPECIAL[name]
    return {'name':name,'label':label,'unit':'','group':group,'type':'category',
            'choices':list(choices),'choice_labels':{str(k):v for k,v in choices.items()},
            'help':HELP.get(name,'Baseline status / medication use: no = 0, yes = 1.')}
