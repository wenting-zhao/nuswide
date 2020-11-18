# nuswide
This repo provides images and labels for NUSWIDE (unofficially). Since the release of NUSWIDE, many original flickr images have gone down. I have collected as many images as possible, and hope this will benefit the community. Any contribution is welcomed! For now, it is only for multi-label classification. NUSWIDE, however, can also be used for image tagging.

A pytorch interface for NUSWIDE classification dataloader is also provided. Used as follows:

```
>>> import nuswide
>>> dataset = nuswide.NUSWIDEClassification('.', 'trainval')
[dataset] read ./classification_labels/classification_trainval.csv
[dataset] NUSWIDE classification set=trainval number of classes=81  number of images=134025
>>> dataset = nuswide.NUSWIDEClassification('.', 'test')
[dataset] read ./classification_labels/classification_test.csv
[dataset] NUSWIDE classification set=test number of classes=81  number of images=89470
```

I downloaded images from https://github.com/thuml/HashNet/tree/master/pytorch. I processed the labels from https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html. Many thanks to them!
