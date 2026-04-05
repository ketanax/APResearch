import json
import matplotlib.pyplot as plt

log_data = """
{"train_lr": 1.0000000000000664e-06, "train_loss": 6.921605627375878, "test_loss": 6.830982051133196, "test_acc1": 0.490000016784668, "test_acc5": 2.0220000640869142, "epoch": 0, "n_parameters": 22050664}
{"train_lr": 1.0000000000000664e-06, "train_loss": 6.883216027101856, "test_loss": 6.782613518594325, "test_acc1": 0.6560000262451172, "test_acc5": 2.5380000942993166, "epoch": 1, "n_parameters": 22050664}
{"train_lr": 2.580000000000285e-05, "train_loss": 6.75490525134502, "test_loss": 6.238929174868996, "test_acc1": 2.7720000929260253, "test_acc5": 9.084000252227783, "epoch": 2, "n_parameters": 22050664}
{"train_lr": 5.060000000000507e-05, "train_loss": 6.581983477840151, "test_loss": 5.875029211300086, "test_acc1": 4.7120001434326175, "test_acc5": 13.834000391235351, "epoch": 3, "n_parameters": 22050664}
{"train_lr": 7.540000000000294e-05, "train_loss": 6.487834637004173, "test_loss": 5.623515995069482, "test_acc1": 6.136000170135498, "test_acc5": 17.566000488433836, "epoch": 4, "n_parameters": 22050664}
{"train_lr": 0.00010019999999999556, "train_loss": 6.408226730487098, "test_loss": 5.435981136628951, "test_acc1": 7.476000240783692, "test_acc5": 20.326000589294434, "epoch": 5, "n_parameters": 22050664}
{"train_lr": 0.0001221857496869664, "train_loss": 6.329198023146647, "test_loss": 5.133405221376383, "test_acc1": 10.444000349731445, "test_acc5": 25.970000655822755, "epoch": 6, "n_parameters": 22050664}
{"train_lr": 0.00012096214793856357, "train_loss": 6.214828687016462, "test_loss": 4.856124349937585, "test_acc1": 13.61600040802002, "test_acc5": 31.182000893859865, "epoch": 7, "n_parameters": 22050664}
{"train_lr": 0.00011952755551680459, "train_loss": 6.1100846352021385, "test_loss": 4.63761874131316, "test_acc1": 15.706000467834473, "test_acc5": 34.8900009072876, "epoch": 8, "n_parameters": 22050664}
{"train_lr": 0.00011788763410252169, "train_loss": 6.017283086932753, "test_loss": 4.3949045564023015, "test_acc1": 19.23600057571411, "test_acc5": 40.14800103424072, "epoch": 9, "n_parameters": 22050664}
{"train_lr": 0.00011604885571636633, "train_loss": 5.919027323225524, "test_loss": 4.141391475995381, "test_acc1": 22.188000667266845, "test_acc5": 44.33200122375488, "epoch": 10, "n_parameters": 22050664}
{"train_lr": 0.00011401847717656128, "train_loss": 5.818013896825123, "test_loss": 4.014967201313296, "test_acc1": 23.940000743103028, "test_acc5": 46.538001341552736, "epoch": 11, "n_parameters": 22050664}
{"train_lr": 0.0001118045114596049, "train_loss": 5.74374088168388, "test_loss": 3.8101617376466366, "test_acc1": 26.696000688476563, "test_acc5": 50.12800152954102, "epoch": 12, "n_parameters": 22050664}
{"train_lr": 0.00010941569607673755, "train_loss": 5.669707585551256, "test_loss": 3.6606588222057885, "test_acc1": 28.804000775146484, "test_acc5": 52.94600153320312, "epoch": 13, "n_parameters": 22050664}
{"train_lr": 0.00010686145859089867, "train_loss": 5.592074125676067, "test_loss": 3.518928346962764, "test_acc1": 30.948000799560546, "test_acc5": 55.650001767578125, "epoch": 14, "n_parameters": 22050664}
{"train_lr": 0.00010415187941054479, "train_loss": 5.539532627893616, "test_loss": 3.40018922570108, "test_acc1": 32.730000942382816, "test_acc5": 57.61000173828125, "epoch": 15, "n_parameters": 22050664}
{"train_lr": 0.00010129765200682623, "train_loss": 5.475736509022781, "test_loss": 3.2855386946393157, "test_acc1": 34.560000990600585, "test_acc5": 59.58200176269531, "epoch": 16, "n_parameters": 22050664}
{"train_lr": 9.831004071130003e-05, "train_loss": 5.423356810669226, "test_loss": 3.212715120836236, "test_acc1": 35.63000094177246, "test_acc5": 60.91400174560547, "epoch": 17, "n_parameters": 22050664}
{"train_lr": 9.520083626085304e-05, "train_loss": 5.350330467458152, "test_loss": 3.0869530012324393, "test_acc1": 37.49400116882324, "test_acc5": 62.85600169921875, "epoch": 18, "n_parameters": 22050664}
{"train_lr": 9.19823092649929e-05, "train_loss": 5.323235507050419, "test_loss": 3.041831573992397, "test_acc1": 38.38000108154297, "test_acc5": 63.63400209716797, "epoch": 19, "n_parameters": 22050664}
{"train_lr": 8.866716177936643e-05, "train_loss": 5.268941771715941, "test_loss": 2.9838369778746388, "test_acc1": 39.47200125, "test_acc5": 64.90000205810547, "epoch": 20, "n_parameters": 22050664}
{"train_lr": 8.526847717655169e-05, "train_loss": 5.2222872131928835, "test_loss": 2.9228770511360462, "test_acc1": 40.31200106689453, "test_acc5": 65.59200191894531, "epoch": 21, "n_parameters": 22050664}
{"train_lr": 8.179966851197534e-05, "train_loss": 5.18040954115932, "test_loss": 2.808194542524915, "test_acc1": 42.018001151123045, "test_acc5": 67.3420021484375, "epoch": 22, "n_parameters": 22050664}
{"train_lr": 7.827442558867832e-05, "train_loss": 5.152948135832336, "test_loss": 2.7733968482620415, "test_acc1": 43.000001198730466, "test_acc5": 68.16800207519532, "epoch": 23, "n_parameters": 22050664}
{"train_lr": 7.470666092995352e-05, "train_loss": 5.1184251365973665, "test_loss": 2.7161076301815865, "test_acc1": 43.9280012109375, "test_acc5": 69.08400189453126, "epoch": 24, "n_parameters": 22050664}
{"train_lr": 7.111045487293919e-05, "train_loss": 5.063815413662261, "test_loss": 2.675207662399701, "test_acc1": 44.388001242675784, "test_acc5": 69.99400208984375, "epoch": 25, "n_parameters": 22050664}
{"train_lr": 6.749999999999836e-05, "train_loss": 5.024621665404618, "test_loss": 2.6044098785097116, "test_acc1": 45.75800127685547, "test_acc5": 70.84800220214844, "epoch": 26, "n_parameters": 22050664}
{"train_lr": 6.388954512706039e-05, "train_loss": 5.008401310907064, "test_loss": 2.573990738939965, "test_acc1": 46.162001372070314, "test_acc5": 71.38200219238281, "epoch": 27, "n_parameters": 22050664}
{"train_lr": 6.029333907004475e-05, "train_loss": 4.955709132582619, "test_loss": 2.5337677540907, "test_acc1": 47.28000133544922, "test_acc5": 71.98600217285156, "epoch": 28, "n_parameters": 22050664}
{"train_lr": 5.672557441132566e-05, "train_loss": 4.916435272952043, "test_loss": 2.4826660510223943, "test_acc1": 48.21800134765625, "test_acc5": 72.88600248535157, "epoch": 29, "n_parameters": 22050664}
{"train_lr": 5.320033148802805e-05, "train_loss": 4.9090074041869745, "test_loss": 2.4350354525321287, "test_acc1": 48.7120012109375, "test_acc5": 73.38800220703125, "epoch": 30, "n_parameters": 22050664}
{"train_lr": 4.973152282344307e-05, "train_loss": 4.882498766698233, "test_loss": 2.4001733738343836, "test_acc1": 49.67200145263672, "test_acc5": 74.0740022314453, "epoch": 31, "n_parameters": 22050664}
{"train_lr": 4.633283822063464e-05, "train_loss": 4.8453120328166, "test_loss": 2.359329715085669, "test_acc1": 50.01600133789063, "test_acc5": 74.62400239746094, "epoch": 32, "n_parameters": 22050664}
{"train_lr": 4.301769073500495e-05, "train_loss": 4.83842089951404, "test_loss": 2.356798343959896, "test_acc1": 50.50000141845703, "test_acc5": 74.87000236816407, "epoch": 33, "n_parameters": 22050664}
{"train_lr": 3.979916373914877e-05, "train_loss": 4.809775019618387, "test_loss": 2.304393499061979, "test_acc1": 51.058001462402345, "test_acc5": 75.54400211425781, "epoch": 34, "n_parameters": 22050664}
{"train_lr": 3.668995928870987e-05, "train_loss": 4.782184490933252, "test_loss": 2.2898145941482193, "test_acc1": 51.726001513671875, "test_acc5": 75.93400241210938, "epoch": 35, "n_parameters": 22050664}
{"train_lr": 3.3702347993184416e-05, "train_loss": 4.7440448782926685, "test_loss": 2.260687717989487, "test_acc1": 51.982001462402344, "test_acc5": 76.31600243164063, "epoch": 36, "n_parameters": 22050664}
{"train_lr": 3.084812058945277e-05, "train_loss": 4.713461567287796, "test_loss": 2.227219160945936, "test_acc1": 52.60200153076172, "test_acc5": 76.77800229980468, "epoch": 37, "n_parameters": 22050664}
{"train_lr": 2.8138541409098066e-05, "train_loss": 4.7082726870090195, "test_loss": 2.2269159862593217, "test_acc1": 52.918001352539065, "test_acc5": 76.92800238769532, "epoch": 38, "n_parameters": 22050664}
{"train_lr": 2.5584303923269948e-05, "train_loss": 4.670201130244873, "test_loss": 2.201004445552826, "test_acc1": 53.26000144042969, "test_acc5": 77.34400265625, "epoch": 39, "n_parameters": 22050664}
{"train_lr": 2.319548854039232e-05, "train_loss": 4.66715072478741, "test_loss": 2.190464747134753, "test_acc1": 53.570001533203126, "test_acc5": 77.51200243652343, "epoch": 40, "n_parameters": 22050664}
{"train_lr": 2.0981522823440565e-05, "train_loss": 4.647980918962287, "test_loss": 2.1548510963432634, "test_acc1": 53.93000152832031, "test_acc5": 77.93800242675782, "epoch": 41, "n_parameters": 22050664}
{"train_lr": 1.8951144283633454e-05, "train_loss": 4.645488955443135, "test_loss": 2.147518239030436, "test_acc1": 54.29000156982422, "test_acc5": 78.11000239257812, "epoch": 42, "n_parameters": 22050664}
{"train_lr": 1.7112365897478584e-05, "train_loss": 4.615683753602587, "test_loss": 2.1230409204731497, "test_acc1": 54.600001640625, "test_acc5": 78.45800225097656, "epoch": 43, "n_parameters": 22050664}
{"train_lr": 1.5472444483203018e-05, "train_loss": 4.600483643155401, "test_loss": 2.108191699817263, "test_acc1": 55.048001520996095, "test_acc5": 78.71200254882812, "epoch": 44, "n_parameters": 22050664}
{"train_lr": 1.4037852061424986e-05, "train_loss": 4.5871863193063405, "test_loss": 2.097942201570533, "test_acc1": 55.12800172363281, "test_acc5": 78.74400220214844, "epoch": 45, "n_parameters": 22050664}
{"train_lr": 1.2814250313027573e-05, "train_loss": 4.583312413297548, "test_loss": 2.088476867968095, "test_acc1": 55.350001430664065, "test_acc5": 78.86400227539062, "epoch": 46, "n_parameters": 22050664}
{"train_lr": 1.1806468235104778e-05, "train_loss": 4.581461397611778, "test_loss": 2.0760393530929684, "test_acc1": 55.530001499023435, "test_acc5": 79.21000234375, "epoch": 47, "n_parameters": 22050664}
{"train_lr": 1.1018483083101198e-05, "train_loss": 4.554597997616649, "test_loss": 2.0658065420914427, "test_acc1": 55.70800151855469, "test_acc5": 79.23200265625, "epoch": 48, "n_parameters": 22050664}
{"train_lr": 1.0453404674417904e-05, "train_loss": 4.542667407882238, "test_loss": 2.054167945608782, "test_acc1": 56.00400161376953, "test_acc5": 79.41600235839844, "epoch": 49, "n_parameters": 22050664}
"""

train_loss = []
val_loss = []
epochs = []

for line in log_data.strip().split("\n"):
    data = json.loads(line)
    epochs.append(data["epoch"])
    train_loss.append(data["train_loss"])
    val_loss.append(data["test_loss"])

plt.figure(figsize=(10,6))

# Training loss (line + circle markers)
plt.plot(
    epochs,
    train_loss,
    linestyle='-',
    marker='o',
    linewidth=2,
    markersize=6,
    label="Training Loss"
)

# Validation loss (line + square markers)
plt.plot(
    epochs,
    val_loss,
    linestyle='-',
    marker='s',
    linewidth=2,
    markersize=6,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss per Epoch")
plt.legend()
plt.grid(True)

plt.show()