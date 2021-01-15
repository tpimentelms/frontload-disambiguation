import matplotlib as mpl
# import matplotlib.pyplot as plt
import seaborn as sns


ASPECT = {
    # 'size': 6.5,
    'font_scale': 2.5,
    'labels': False,
    'name_suffix': 'small__shared',
    # 'ratio': 1.625,
    'ratio': 1.625 / 1.5,
    'width': 6.5 * 1.625,
}
ASPECT['height'] = ASPECT['width'] / ASPECT['ratio']

sns.color_palette("PRGn", 14)
sns.set(style="white", palette="muted", color_codes=True)
sns.set_context("notebook", font_scale=ASPECT['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
mpl.rc('figure', figsize=(ASPECT['width'], ASPECT['height']))
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
