# VEGAN
### KUNSTlig intelligens

Clone of [Zach Monge](https://github.com/zachmonge)'s [Forest GAN](https://github.com/zachmonge/cyclegan_forest_abstract_art_Duke_zm), cleaned up to accept updates in various Python packages and [PyTorch Cycle-GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

### Abstract Corals
<p float="left">
  <img src="github_example_images/14.png" width="200" />
  <img src="github_example_images/22.png" width="200" /> 
  <img src="github_example_images/27.png" width="200" />
  <img src="github_example_images/48.png" width="200" />
  <img src="github_example_images/61.png" width="200" />
  <img src="github_example_images/90.png" width="200" /> 
  <img src="github_example_images/94.png" width="200" />
  <img src="github_example_images/167.png" width="200" />
  <img src="github_example_images/184.png" width="200" />
  <img src="github_example_images/191.png" width="200" /> 
  <img src="github_example_images/257.png" width="200" />
  <img src="github_example_images/263.png" width="200" />
</p>

### Structure
```
.
├── cgan
│   ├── data
│   │   ├── aligned_dataset.py
│   │   ├── base_data_loader.py
│   │   ├── base_dataset.py
│   │   ├── custom_dataset_data_loader.py
│   │   ├── data_loader.py
│   │   ├── image_folder.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── data_loader.cpython-36.pyc
│   │   │   └── __init__.cpython-36.pyc
│   │   ├── single_dataset.py
│   │   └── unaligned_dataset.py
│   ├── imgs
│   │   ├── edges2cats.jpg
│   │   └── horse2zebra.gif
│   ├── LICENSE
│   ├── models
│   │   ├── base_model.py
│   │   ├── cycle_gan_model.py
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── networks.py
│   │   ├── pix2pix_model.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-36.pyc
│   │   │   └── models.cpython-36.pyc
│   │   └── test_model.py
│   ├── options
│   │   ├── base_options.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── base_options.cpython-36.pyc
│   │   │   ├── __init__.cpython-36.pyc
│   │   │   └── train_options.cpython-36.pyc
│   │   ├── test_options.py
│   │   └── train_options.py
│   ├── pretrained_models
│   │   ├── download_cyclegan_model.sh
│   │   └── download_pix2pix_model.sh
│   ├── README.md
│   ├── scripts
│   │   ├── test_cyclegan.sh
│   │   ├── test_pix2pix.sh
│   │   ├── test_single.sh
│   │   ├── train_cyclegan.sh
│   │   └── train_pix2pix.sh
│   ├── test.py
│   ├── train.py
│   └── util
│       ├── get_data.py
│       ├── html.py
│       ├── image_pool.py
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── html.cpython-36.pyc
│       │   ├── __init__.cpython-36.pyc
│       │   ├── util.cpython-36.pyc
│       │   └── visualizer.cpython-36.pyc
│       ├── util.py
│       └── visualizer.py
├── cyclegan_forest_abstract_art_Duke_zm.ipynb
├── environment.yml
├── fastai-master
│   ├── AUTHORS.md
│   ├── azure-pipelines.yml
│   ├── builds
│   │   ├── custom-conda-builds
│   │   │   ├── dataclasses
│   │   │   │   └── meta.yaml
│   │   │   ├── libjpeg-turbo
│   │   │   │   ├── conda-build.txt
│   │   │   │   └── recipe
│   │   │   │       ├── bld.bat
│   │   │   │       ├── build.sh
│   │   │   │       ├── LICENSE.md
│   │   │   │       ├── meta.yaml
│   │   │   │       └── testorig.jpg
│   │   │   ├── libtiff
│   │   │   │   ├── conda-build.txt
│   │   │   │   └── recipe
│   │   │   │       ├── 0001-CVE-2017-11613_part1.patch
│   │   │   │       ├── 0002-CVE-2017-11613_part2.patch
│   │   │   │       ├── bld.bat
│   │   │   │       ├── build.sh
│   │   │   │       └── meta.yaml
│   │   │   ├── nvidia-ml-py3
│   │   │   │   └── meta.yaml
│   │   │   ├── pillow
│   │   │   │   ├── conda-build.txt
│   │   │   │   └── recipe
│   │   │   │       ├── bld.bat
│   │   │   │       ├── build.sh
│   │   │   │       └── meta.yaml
│   │   │   ├── pillow-simd
│   │   │   │   ├── conda-build.txt
│   │   │   │   ├── libjpeg-turbo-feature.patch
│   │   │   │   ├── pretend-pill-package.patch
│   │   │   │   └── recipe
│   │   │   │       ├── bld.bat
│   │   │   │       ├── build.sh
│   │   │   │       └── meta.yaml
│   │   │   ├── plac
│   │   │   │   └── meta.yaml
│   │   │   ├── README.md
│   │   │   ├── torchvision
│   │   │   │   └── meta.yaml
│   │   │   └── torchvision-cpu
│   │   │       └── meta.yaml
│   │   └── custom-pip-builds
│   │       └── torchvision
│   │           ├── README.md
│   │           └── setup.py
│   ├── CHANGES.md
│   ├── cloudbuild.yaml
│   ├── CODE-OF-CONDUCT.md
│   ├── conda
│   │   └── meta.yaml
│   ├── CONTRIBUTING.md
│   ├── courses
│   │   ├── 00-DO-NOT-USE-WITH-FASTAI-1.0.x.txt
│   │   ├── dl1
│   │   │   ├── 00-DO-NOT-USE-WITH-FASTAI-1.0.x.txt
│   │   │   ├── adamw-sgdw-demo.ipynb
│   │   │   ├── cifar10.ipynb
│   │   │   ├── cifar10-simplenet.ipynb
│   │   │   ├── embedding_refactoring_unit_tests.ipynb
│   │   │   ├── excel
│   │   │   │   ├── collab_filter.xlsx
│   │   │   │   ├── conv-example.xlsx
│   │   │   │   ├── entropy_example.xlsx
│   │   │   │   ├── graddesc.xlsm
│   │   │   │   └── layers_example.xlsx
│   │   │   ├── fastai
│   │   │   ├── fish.ipynb
│   │   │   ├── images
│   │   │   │   ├── pretrained.png
│   │   │   │   ├── sgdr.png
│   │   │   │   ├── zeiler1.png
│   │   │   │   ├── zeiler2.png
│   │   │   │   ├── zeiler3.png
│   │   │   │   └── zeiler4.png
│   │   │   ├── keras_lesson1.ipynb
│   │   │   ├── lang_model-arxiv.ipynb
│   │   │   ├── lang_model.ipynb
│   │   │   ├── lesson1-breeds.ipynb
│   │   │   ├── lesson1.ipynb
│   │   │   ├── lesson1-rxt50.ipynb
│   │   │   ├── lesson1-vgg.ipynb
│   │   │   ├── lesson2-image_models.ipynb
│   │   │   ├── lesson3-rossman.ipynb
│   │   │   ├── lesson4-imdb.ipynb
│   │   │   ├── lesson5-movielens.ipynb
│   │   │   ├── lesson6-rnn.ipynb
│   │   │   ├── lesson6-sgd.ipynb
│   │   │   ├── lesson7-CAM.ipynb
│   │   │   ├── lesson7-cifar10.ipynb
│   │   │   ├── nasnet.ipynb
│   │   │   ├── nlp-arxiv.ipynb
│   │   │   ├── nlp.ipynb
│   │   │   ├── planet_cv.ipynb
│   │   │   ├── planet.py
│   │   │   ├── ppt
│   │   │   │   └── lesson6.pptx
│   │   │   ├── rossman_exp.py
│   │   │   ├── scripts
│   │   │   │   └── train_planet.py
│   │   │   ├── test_transforms.ipynb
│   │   │   └── xor.ipynb
│   │   ├── dl2
│   │   │   ├── 00-DO-NOT-USE-WITH-FASTAI-1.0.x.txt
│   │   │   ├── carvana.ipynb
│   │   │   ├── carvana-unet.ipynb
│   │   │   ├── carvana-unet-lrg.ipynb
│   │   │   ├── cgan
│   │   │   │   ├── data
│   │   │   │   │   ├── aligned_dataset.py
│   │   │   │   │   ├── base_data_loader.py
│   │   │   │   │   ├── base_dataset.py
│   │   │   │   │   ├── custom_dataset_data_loader.py
│   │   │   │   │   ├── data_loader.py
│   │   │   │   │   ├── image_folder.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── single_dataset.py
│   │   │   │   │   └── unaligned_dataset.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── models
│   │   │   │   │   ├── base_model.py
│   │   │   │   │   ├── cycle_gan_model.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── models.py
│   │   │   │   │   ├── networks.py
│   │   │   │   │   ├── pix2pix_model.py
│   │   │   │   │   └── test_model.py
│   │   │   │   ├── options
│   │   │   │   │   ├── base_options.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── test_options.py
│   │   │   │   │   └── train_options.py
│   │   │   │   ├── test.py
│   │   │   │   ├── train.py
│   │   │   │   └── util
│   │   │   │       ├── get_data.py
│   │   │   │       ├── html.py
│   │   │   │       ├── image_pool.py
│   │   │   │       ├── __init__.py
│   │   │   │       ├── util.py
│   │   │   │       └── visualizer.py
│   │   │   ├── cifar10-darknet.ipynb
│   │   │   ├── cifar10-dawn.ipynb
│   │   │   ├── cyclegan.ipynb
│   │   │   ├── devise.ipynb
│   │   │   ├── enhance.ipynb
│   │   │   ├── fastai
│   │   │   ├── imdb.ipynb
│   │   │   ├── imdb_scripts
│   │   │   │   ├── create_toks.py
│   │   │   │   ├── eval_clas.py
│   │   │   │   ├── finetune_lm.py
│   │   │   │   ├── lr_plot.png
│   │   │   │   ├── merge_wiki.py
│   │   │   │   ├── predict_with_classifier.py
│   │   │   │   ├── prepare_wiki.sh
│   │   │   │   ├── pretrain_lm.py
│   │   │   │   ├── README.md
│   │   │   │   ├── sampled_sm.py
│   │   │   │   ├── tok2id.py
│   │   │   │   └── train_clas.py
│   │   │   ├── lsun_scripts
│   │   │   │   ├── lsun-data.py
│   │   │   │   └── lsun-download.py
│   │   │   ├── pascal.ipynb
│   │   │   ├── pascal-multi.ipynb
│   │   │   ├── ppt
│   │   │   │   └── lesson8.pptx
│   │   │   ├── style-transfer.ipynb
│   │   │   ├── style-transfer-net.ipynb
│   │   │   ├── training_phase.ipynb
│   │   │   ├── translate.ipynb
│   │   │   ├── wgan.ipynb
│   │   │   └── xl
│   │   │       └── dl-examples.xlsx
│   │   └── ml1
│   │       ├── 00-DO-NOT-USE-WITH-FASTAI-1.0.x.txt
│   │       ├── bulldozer_dl.ipynb
│   │       ├── bulldozer_linreg.ipynb
│   │       ├── Ethics in Data Science.ipynb
│   │       ├── excel
│   │       │   └── naivebayes.xlsx
│   │       ├── fastai
│   │       ├── images
│   │       │   ├── bulldozers_data2.png
│   │       │   ├── bulldozers_data.png
│   │       │   ├── digit.gif
│   │       │   ├── ethics_recidivism.jpg
│   │       │   ├── mnist.png
│   │       │   ├── overfitting2.png
│   │       │   ├── sgd2.gif
│   │       │   ├── what_is_pytorch.png
│   │       │   ├── zeiler1.png
│   │       │   ├── zeiler2.png
│   │       │   ├── zeiler3.png
│   │       │   └── zeiler4.png
│   │       ├── lesson1-rf.ipynb
│   │       ├── lesson2-rf_interpretation.ipynb
│   │       ├── lesson3-rf_foundations.ipynb
│   │       ├── lesson4-mnist_sgd.ipynb
│   │       ├── lesson5-nlp.ipynb
│   │       └── ppt
│   │           ├── 2017-12-ethics.pptx
│   │           └── ml_applications.pptx
│   ├── docs
│   │   ├── 404.md
│   │   ├── applications.html
│   │   ├── basic_data.html
│   │   ├── basic_train.html
│   │   ├── callback.html
│   │   ├── callbacks.csv_logger.html
│   │   ├── callbacks.fp16.html
│   │   ├── callbacks.general_sched.html
│   │   ├── callbacks.hooks.html
│   │   ├── callbacks.html
│   │   ├── callbacks.lr_finder.html
│   │   ├── callbacks.mem.html
│   │   ├── callbacks.misc.html
│   │   ├── callbacks.mixup.html
│   │   ├── callbacks.one_cycle.html
│   │   ├── callbacks.rnn.html
│   │   ├── callbacks.tensorboard.html
│   │   ├── callbacks.tracker.html
│   │   ├── CNAME
│   │   ├── collab.html
│   │   ├── _config.yml
│   │   ├── core.html
│   │   ├── createtag
│   │   ├── css
│   │   │   ├── bootstrap.min.css
│   │   │   ├── boxshadowproperties.css
│   │   │   ├── customstyles.css
│   │   │   ├── font-awesome.min.css
│   │   │   ├── fonts
│   │   │   │   ├── FontAwesome.otf
│   │   │   │   ├── fontawesome-webfont.eot
│   │   │   │   ├── fontawesome-webfont.svg
│   │   │   │   ├── fontawesome-webfont.ttf
│   │   │   │   ├── fontawesome-webfont.woff
│   │   │   │   └── fontawesome-webfont.woff2
│   │   │   ├── modern-business.css
│   │   │   ├── printstyles.css
│   │   │   ├── syntax.css
│   │   │   ├── theme-blue.css
│   │   │   └── theme-green.css
│   │   ├── _data
│   │   │   ├── alerts.yml
│   │   │   ├── definitions.yml
│   │   │   ├── glossary.yml
│   │   │   ├── samplelist.yml
│   │   │   ├── sidebars
│   │   │   │   ├── home_sidebar.yml
│   │   │   │   ├── mydoc_sidebar.yml
│   │   │   │   ├── other.yml
│   │   │   │   ├── product1_sidebar.yml
│   │   │   │   └── product2_sidebar.yml
│   │   │   ├── strings.yml
│   │   │   ├── tags.yml
│   │   │   ├── terms.yml
│   │   │   └── topnav.yml
│   │   ├── data_block.html
│   │   ├── datasets.html
│   │   ├── dev
│   │   │   ├── abbr.md
│   │   │   ├── develop.md
│   │   │   ├── git.md
│   │   │   ├── gpu.md
│   │   │   ├── index.md
│   │   │   ├── release.md
│   │   │   ├── style.md
│   │   │   └── test.md
│   │   ├── distributed.md
│   │   ├── docs-build.txt
│   │   ├── fastai_typing.html
│   │   ├── feed.xml
│   │   ├── fonts
│   │   │   ├── FontAwesome.otf
│   │   │   ├── fontawesome-webfont.eot
│   │   │   ├── fontawesome-webfont.svg
│   │   │   ├── fontawesome-webfont.ttf
│   │   │   ├── fontawesome-webfont.woff
│   │   │   ├── glyphicons-halflings-regular.eot
│   │   │   ├── glyphicons-halflings-regular.svg
│   │   │   ├── glyphicons-halflings-regular.ttf
│   │   │   ├── glyphicons-halflings-regular.woff
│   │   │   └── glyphicons-halflings-regular.woff2
│   │   ├── Gemfile
│   │   ├── Gemfile.lock
│   │   ├── gen_doc.convert2html.html
│   │   ├── gen_doc.gen_notebooks.html
│   │   ├── gen_doc.html
│   │   ├── gen_doc_main.md
│   │   ├── gen_doc.nbdoc.html
│   │   ├── gen_doc.nbtest.html
│   │   ├── images
│   │   │   ├── androidsdkmanagericon.png
│   │   │   ├── authorizegithubscreen2.png
│   │   │   ├── authorizeongithub.png
│   │   │   ├── company_logo_big.png
│   │   │   ├── company_logo.png
│   │   │   ├── favicon.ico
│   │   │   ├── helpapi-01.png
│   │   │   ├── helpapi.svg
│   │   │   ├── illustratoroptions.png
│   │   │   ├── itermexample.png
│   │   │   ├── jekyll.png
│   │   │   ├── killalljekyll.png
│   │   │   ├── liningup.png
│   │   │   └── workflowarrow.png
│   │   ├── imgs
│   │   │   ├── betadist-mixup.png
│   │   │   ├── button_hide.png
│   │   │   ├── car_bbox.jpg
│   │   │   ├── car_example.jpg
│   │   │   ├── cat_example.jpg
│   │   │   ├── dependencies.svg
│   │   │   ├── face_example.jpg
│   │   │   ├── fastai_full.svg
│   │   │   ├── fastai_internal.svg
│   │   │   ├── from_folder.png
│   │   │   ├── grid.png
│   │   │   ├── grid_rot.png
│   │   │   ├── hide_input.png
│   │   │   ├── mask_example.png
│   │   │   ├── mask_rle_sample.csv
│   │   │   ├── mix_match.png
│   │   │   ├── mixup.png
│   │   │   ├── nbext.png
│   │   │   ├── onecycle_finder.png
│   │   │   ├── onecycle_params.png
│   │   │   ├── one_interpol.png
│   │   │   ├── tfm_bbox.png
│   │   │   ├── tfm_mask.png
│   │   │   ├── train_graph.gif
│   │   │   ├── two_interpol.png
│   │   │   └── u-net-architecture.png
│   │   ├── imports.md
│   │   ├── _includes
│   │   │   ├── archive.html
│   │   │   ├── callout.html
│   │   │   ├── custom
│   │   │   │   ├── getting_started_series.html
│   │   │   │   ├── getting_started_series_next.html
│   │   │   │   ├── series_acme.html
│   │   │   │   ├── series_acme_next.html
│   │   │   │   ├── usermapcomplex.html
│   │   │   │   └── usermap.html
│   │   │   ├── disqus.html
│   │   │   ├── feedback.html
│   │   │   ├── footer.html
│   │   │   ├── google_analytics.html
│   │   │   ├── head.html
│   │   │   ├── head_print.html
│   │   │   ├── image.html
│   │   │   ├── important.html
│   │   │   ├── initialize_shuffle.html
│   │   │   ├── inline_image.html
│   │   │   ├── links.html
│   │   │   ├── note.html
│   │   │   ├── search_google_custom.html
│   │   │   ├── search_simple_jekyll.html
│   │   │   ├── sidebar.html
│   │   │   ├── taglogic.html
│   │   │   ├── tip.html
│   │   │   ├── toc.html
│   │   │   ├── topnav.html
│   │   │   └── warning.html
│   │   ├── index.html
│   │   ├── install.md
│   │   ├── jekyll_metadata.html
│   │   ├── js
│   │   │   ├── customscripts.js
│   │   │   ├── jekyll-search.js
│   │   │   ├── jquery.ba-throttle-debounce.min.js
│   │   │   ├── jquery.navgoco.min.js
│   │   │   ├── jquery.shuffle.min.js
│   │   │   └── toc.js
│   │   ├── layers.html
│   │   ├── _layouts
│   │   │   ├── default.html
│   │   │   ├── default_print.html
│   │   │   ├── none.html
│   │   │   ├── page.html
│   │   │   ├── page_print.html
│   │   │   └── post.html
│   │   ├── licenses
│   │   │   ├── LICENSE
│   │   │   └── LICENSE-BSD-NAVGOCO.txt
│   │   ├── metrics.html
│   │   ├── overview.html
│   │   ├── pdfconfigs
│   │   │   ├── config_mydoc_pdf.yml
│   │   │   ├── config_product1_pdf.yml
│   │   │   ├── config_product2_pdf.yml
│   │   │   ├── prince-list.txt
│   │   │   ├── titlepage.html
│   │   │   └── tocpage.html
│   │   ├── pdf-mydoc.sh
│   │   ├── performance.md
│   │   ├── s3_website.yml
│   │   ├── search.json
│   │   ├── sitemap.xml
│   │   ├── support.md
│   │   ├── tabular.data.html
│   │   ├── tabular.html
│   │   ├── tabular.models.html
│   │   ├── tabular.transform.html
│   │   ├── text.data.html
│   │   ├── text.html
│   │   ├── text.interpret.html
│   │   ├── text.learner.html
│   │   ├── text.models.html
│   │   ├── text.transform.html
│   │   ├── _tooltips
│   │   │   ├── baseball.html
│   │   │   ├── basketball.html
│   │   │   ├── football.html
│   │   │   └── soccer.html
│   │   ├── tooltips.json
│   │   ├── torch_core.html
│   │   ├── train.html
│   │   ├── training.html
│   │   ├── troubleshoot.md
│   │   ├── tutorial.data.html
│   │   ├── tutorial.inference.html
│   │   ├── tutorial.itemlist.html
│   │   ├── tutorial.resources.md
│   │   ├── tutorials.md
│   │   ├── update.sh
│   │   ├── utils.collect_env.html
│   │   ├── utils.ipython.html
│   │   ├── utils.mem.html
│   │   ├── utils.mod_display.html
│   │   ├── vision.data.html
│   │   ├── vision.gan.html
│   │   ├── vision.html
│   │   ├── vision.image.html
│   │   ├── vision.interpret.html
│   │   ├── vision.learner.html
│   │   ├── vision.models.html
│   │   ├── vision.models.unet.html
│   │   ├── vision.transform.html
│   │   ├── widgets.class_confusion.html
│   │   └── widgets.image_cleaner.html
│   ├── docs_src
│   │   ├── applications.ipynb
│   │   ├── basic_data.ipynb
│   │   ├── basic_train.ipynb
│   │   ├── callback.ipynb
│   │   ├── callbacks.csv_logger.ipynb
│   │   ├── callbacks.fp16.ipynb
│   │   ├── callbacks.general_sched.ipynb
│   │   ├── callbacks.hooks.ipynb
│   │   ├── callbacks.ipynb
│   │   ├── callbacks.lr_finder.ipynb
│   │   ├── callbacks.mem.ipynb
│   │   ├── callbacks.misc.ipynb
│   │   ├── callbacks.mixup.ipynb
│   │   ├── callbacks.one_cycle.ipynb
│   │   ├── callbacks.rnn.ipynb
│   │   ├── callbacks.tensorboard.ipynb
│   │   ├── callbacks.tracker.ipynb
│   │   ├── collab.ipynb
│   │   ├── conftest.py
│   │   ├── core.ipynb
│   │   ├── data_block.ipynb
│   │   ├── datasets.ipynb
│   │   ├── fastai_typing.ipynb
│   │   ├── gen_doc.convert2html.ipynb
│   │   ├── gen_doc.gen_notebooks.ipynb
│   │   ├── gen_doc.ipynb
│   │   ├── gen_doc.nbdoc.ipynb
│   │   ├── gen_doc.nbtest.ipynb
│   │   ├── imgs
│   │   ├── index.ipynb
│   │   ├── jekyll_metadata.ipynb
│   │   ├── js
│   │   │   └── toc.js
│   │   ├── layers.ipynb
│   │   ├── metrics.ipynb
│   │   ├── nbval
│   │   │   ├── cover.py
│   │   │   ├── __init__.py
│   │   │   ├── kernel.py
│   │   │   ├── nbdime_reporter.py
│   │   │   ├── plugin.py
│   │   │   ├── README.md
│   │   │   └── _version.py
│   │   ├── overview.ipynb
│   │   ├── points.pth
│   │   ├── pytest.ini
│   │   ├── run_tests.sh
│   │   ├── sidebar
│   │   │   └── sidebar_data.py
│   │   ├── tabular.data.ipynb
│   │   ├── tabular.ipynb
│   │   ├── tabular.models.ipynb
│   │   ├── tabular.transform.ipynb
│   │   ├── text.data.ipynb
│   │   ├── text.interpret.ipynb
│   │   ├── text.ipynb
│   │   ├── text.learner.ipynb
│   │   ├── text.models.ipynb
│   │   ├── text.transform.ipynb
│   │   ├── torch_core.ipynb
│   │   ├── training.ipynb
│   │   ├── train.ipynb
│   │   ├── trustnbs.py
│   │   ├── tutorial.data.ipynb
│   │   ├── tutorial.inference.ipynb
│   │   ├── tutorial.itemlist.ipynb
│   │   ├── utils.collect_env.ipynb
│   │   ├── utils.ipython.ipynb
│   │   ├── utils.mem.ipynb
│   │   ├── utils.mod_display.ipynb
│   │   ├── vision.data.ipynb
│   │   ├── vision.gan.ipynb
│   │   ├── vision.image.ipynb
│   │   ├── vision.interpret.ipynb
│   │   ├── vision.ipynb
│   │   ├── vision.learner.ipynb
│   │   ├── vision.models.ipynb
│   │   ├── vision.models.unet.ipynb
│   │   ├── vision.transform.ipynb
│   │   ├── widgets.class_confusion.ipynb
│   │   └── widgets.image_cleaner.ipynb
│   ├── environment-cpu.yml
│   ├── environment.yml
│   ├── examples
│   │   ├── cifar.ipynb
│   │   ├── collab.ipynb
│   │   ├── data
│   │   ├── dogs_cats.ipynb
│   │   ├── README.md
│   │   ├── tabular.ipynb
│   │   ├── text.ipynb
│   │   ├── train_cifar.py
│   │   ├── train_imagenet.py
│   │   ├── train_imagenette_adv.py
│   │   ├── train_imagenette.py
│   │   ├── train_mnist.py
│   │   ├── train_wt103.py
│   │   ├── ULMFit.ipynb
│   │   └── vision.ipynb
│   ├── fastai
│   │   ├── basic_data.py
│   │   ├── basics.py
│   │   ├── basic_train.py
│   │   ├── callback.py
│   │   ├── callbacks
│   │   │   ├── csv_logger.py
│   │   │   ├── fp16.py
│   │   │   ├── general_sched.py
│   │   │   ├── hooks.py
│   │   │   ├── __init__.py
│   │   │   ├── loss_metrics.py
│   │   │   ├── lr_finder.py
│   │   │   ├── mem.py
│   │   │   ├── misc.py
│   │   │   ├── mixup.py
│   │   │   ├── mlflow.py
│   │   │   ├── one_cycle.py
│   │   │   ├── oversampling.py
│   │   │   ├── rnn.py
│   │   │   ├── tensorboard.py
│   │   │   └── tracker.py
│   │   ├── collab.py
│   │   ├── core.py
│   │   ├── data_block.py
│   │   ├── datasets.py
│   │   ├── distributed.py
│   │   ├── gen_doc
│   │   │   ├── autogen.tpl
│   │   │   ├── convert2html.py
│   │   │   ├── core.py
│   │   │   ├── docstrings.py
│   │   │   ├── doctest.py
│   │   │   ├── gen_notebooks.py
│   │   │   ├── hide.tpl
│   │   │   ├── __init__.py
│   │   │   ├── jekyll.tpl
│   │   │   ├── nbdoc.py
│   │   │   └── nbtest.py
│   │   ├── general_optimizer.py
│   │   ├── imports
│   │   │   ├── core.py
│   │   │   ├── __init__.py
│   │   │   └── torch.py
│   │   ├── __init__.py
│   │   ├── launch.py
│   │   ├── layers.py
│   │   ├── metrics.py
│   │   ├── script.py
│   │   ├── sixel.py
│   │   ├── tabular
│   │   │   ├── data.py
│   │   │   ├── __init__.py
│   │   │   ├── models.py
│   │   │   └── transform.py
│   │   ├── test_registry.json
│   │   ├── text
│   │   │   ├── data.py
│   │   │   ├── __init__.py
│   │   │   ├── interpret.py
│   │   │   ├── learner.py
│   │   │   ├── models
│   │   │   │   ├── awd_lstm.py
│   │   │   │   ├── bwd_forget_mult_cuda.cpp
│   │   │   │   ├── bwd_forget_mult_cuda_kernel.cu
│   │   │   │   ├── forget_mult_cuda.cpp
│   │   │   │   ├── forget_mult_cuda_kernel.cu
│   │   │   │   ├── __init__.py
│   │   │   │   ├── qrnn.py
│   │   │   │   └── transformer.py
│   │   │   └── transform.py
│   │   ├── torch_core.py
│   │   ├── train.py
│   │   ├── utils
│   │   │   ├── check_perf.py
│   │   │   ├── collect_env.py
│   │   │   ├── __init__.py
│   │   │   ├── ipython.py
│   │   │   ├── mem.py
│   │   │   ├── mod_display.py
│   │   │   ├── pynvml_gate.py
│   │   │   └── show_install.py
│   │   ├── version.py
│   │   ├── vision
│   │   │   ├── cyclegan.py
│   │   │   ├── data.py
│   │   │   ├── gan.py
│   │   │   ├── image.py
│   │   │   ├── __init__.py
│   │   │   ├── interpret.py
│   │   │   ├── learner.py
│   │   │   ├── models
│   │   │   │   ├── cadene_models.py
│   │   │   │   ├── darknet.py
│   │   │   │   ├── efficientnet.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── presnet.py
│   │   │   │   ├── unet.py
│   │   │   │   ├── wrn.py
│   │   │   │   ├── xception.py
│   │   │   │   ├── xresnet2.py
│   │   │   │   └── xresnet.py
│   │   │   ├── transform.py
│   │   │   └── tta.py
│   │   └── widgets
│   │       ├── class_confusion.py
│   │       ├── image_cleaner.py
│   │       ├── image_downloader.py
│   │       └── __init__.py
│   ├── LICENSE
│   ├── Makefile
│   ├── MANIFEST.in
│   ├── old
│   │   ├── docs
│   │   │   ├── abbr.md
│   │   │   ├── anatomy.adoc
│   │   │   ├── dataloader.adoc
│   │   │   ├── expand_adoc_templ.ipynb
│   │   │   ├── gen_ascii_docs.py
│   │   │   ├── __init__.py
│   │   │   ├── md_expander.py
│   │   │   ├── module-decisions.md
│   │   │   ├── README.md
│   │   │   ├── style.md
│   │   │   ├── tabular_data
│   │   │   ├── templates.py
│   │   │   ├── testing.adoc
│   │   │   ├── transforms.adoc
│   │   │   ├── transforms.html
│   │   │   └── transforms-tmpl.adoc
│   │   ├── fastai
│   │   │   ├── adaptive_softmax.py
│   │   │   ├── column_data.py
│   │   │   ├── conv_learner.py
│   │   │   ├── core.py
│   │   │   ├── dataloader.py
│   │   │   ├── dataset.py
│   │   │   ├── executors.py
│   │   │   ├── fp16.py
│   │   │   ├── images
│   │   │   │   └── industrial_fishing.png
│   │   │   ├── imports.py
│   │   │   ├── initializers.py
│   │   │   ├── __init__.py
│   │   │   ├── io.py
│   │   │   ├── layer_optimizer.py
│   │   │   ├── layers.py
│   │   │   ├── learner.py
│   │   │   ├── lm_rnn.py
│   │   │   ├── lsuv_initializer.py
│   │   │   ├── metrics.py
│   │   │   ├── model.py
│   │   │   ├── models
│   │   │   │   ├── cifar10
│   │   │   │   │   ├── main_dxy.py
│   │   │   │   │   ├── main_kuangliu.py
│   │   │   │   │   ├── main.sh
│   │   │   │   │   ├── preact_resnet.py
│   │   │   │   │   ├── resnext.py
│   │   │   │   │   ├── senet.py
│   │   │   │   │   ├── utils_kuangliu.py
│   │   │   │   │   ├── utils.py
│   │   │   │   │   └── wideresnet.py
│   │   │   │   ├── convert_torch.py
│   │   │   │   ├── darknet.py
│   │   │   │   ├── fa_resnet.py
│   │   │   │   ├── inceptionresnetv2.py
│   │   │   │   ├── inceptionv4.py
│   │   │   │   ├── nasnet.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── fa_resnet.cpython-36.pyc
│   │   │   │   │   ├── inceptionresnetv2.cpython-36.pyc
│   │   │   │   │   ├── inceptionv4.cpython-36.pyc
│   │   │   │   │   ├── nasnet.cpython-36.pyc
│   │   │   │   │   ├── resnext_101_32x4d.cpython-36.pyc
│   │   │   │   │   ├── resnext_101_64x4d.cpython-36.pyc
│   │   │   │   │   ├── resnext_50_32x4d.cpython-36.pyc
│   │   │   │   │   └── wrn_50_2f.cpython-36.pyc
│   │   │   │   ├── resnet.py
│   │   │   │   ├── resnext_101_32x4d.py
│   │   │   │   ├── resnext_101_64x4d.py
│   │   │   │   ├── resnext_50_32x4d.py
│   │   │   │   ├── senet.py
│   │   │   │   ├── unet.py
│   │   │   │   ├── wideresnet.py
│   │   │   │   └── wrn_50_2f.py
│   │   │   ├── nlp.py
│   │   │   ├── plots.py
│   │   │   ├── __pycache__
│   │   │   │   ├── conv_learner.cpython-36.pyc
│   │   │   │   ├── core.cpython-36.pyc
│   │   │   │   ├── dataloader.cpython-36.pyc
│   │   │   │   ├── dataset.cpython-36.pyc
│   │   │   │   ├── fp16.cpython-36.pyc
│   │   │   │   ├── imports.cpython-36.pyc
│   │   │   │   ├── __init__.cpython-36.pyc
│   │   │   │   ├── initializers.cpython-36.pyc
│   │   │   │   ├── layer_optimizer.cpython-36.pyc
│   │   │   │   ├── layers.cpython-36.pyc
│   │   │   │   ├── learner.cpython-36.pyc
│   │   │   │   ├── lsuv_initializer.cpython-36.pyc
│   │   │   │   ├── metrics.cpython-36.pyc
│   │   │   │   ├── model.cpython-36.pyc
│   │   │   │   ├── sgdr.cpython-36.pyc
│   │   │   │   ├── swa.cpython-36.pyc
│   │   │   │   ├── torch_imports.cpython-36.pyc
│   │   │   │   └── transforms.cpython-36.pyc
│   │   │   ├── rnn_reg.py
│   │   │   ├── rnn_train.py
│   │   │   ├── set_spawn.py
│   │   │   ├── sgdr.py
│   │   │   ├── structured.py
│   │   │   ├── swa.py
│   │   │   ├── text.py
│   │   │   ├── torch_imports.py
│   │   │   ├── torchqrnn
│   │   │   │   ├── forget_mult.py
│   │   │   │   └── qrnn.py
│   │   │   ├── transforms_pil.py
│   │   │   └── transforms.py
│   │   ├── LICENSE
│   │   ├── MANIFEST.in
│   │   ├── pytest.ini
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── setup.cfg
│   │   ├── setup.py
│   │   ├── tests
│   │   │   ├── __init__.py
│   │   │   ├── test_core.py
│   │   │   ├── test_dataset.py
│   │   │   ├── test_layer_optimizer.py
│   │   │   ├── test_lsuv_initializer.py
│   │   │   ├── test_samplers.py
│   │   │   ├── test_structured.py
│   │   │   └── test_transform.py
│   │   └── tutorials
│   │       ├── fastai
│   │       ├── images
│   │       │   ├── cifar10.png
│   │       │   ├── demba_combustion_engine.png
│   │       │   ├── digit.gif
│   │       │   ├── fashion-mnist.png
│   │       │   ├── markov_health.jpg
│   │       │   ├── mnist.png
│   │       │   ├── normal.jpg
│   │       │   ├── overfitting2.png
│   │       │   ├── overfitting.png
│   │       │   ├── sgd2.gif
│   │       │   ├── shop.png
│   │       │   ├── what_is_pytorch.png
│   │       │   ├── zeiler1.png
│   │       │   ├── zeiler2.png
│   │       │   ├── zeiler3.png
│   │       │   └── zeiler4.png
│   │       ├── __init__.py
│   │       ├── kmeans.py
│   │       ├── linalg_pytorch.ipynb
│   │       └── meanshift.ipynb
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.cfg
│   ├── setup.py
│   ├── tests
│   │   ├── conftest.py
│   │   ├── data
│   │   ├── test_basic_data.py
│   │   ├── test_basic_train.py
│   │   ├── test_batchnom_issue_minimal.py
│   │   ├── test_callback.py
│   │   ├── test_callbacks_csv_logger.py
│   │   ├── test_callbacks_hooks.py
│   │   ├── test_callbacks_mem.py
│   │   ├── test_callbacks_misc.py
│   │   ├── test_collab_train.py
│   │   ├── test_core.py
│   │   ├── test_data_block.py
│   │   ├── test_datasets.py
│   │   ├── test_fp16.py
│   │   ├── test_gen_doc_nbtest.py
│   │   ├── test_layers.py
│   │   ├── test_metrics.py
│   │   ├── test_mod_display.py
│   │   ├── test_tabular_data.py
│   │   ├── test_tabular_train.py
│   │   ├── test_tabular_transform.py
│   │   ├── test_text_data.py
│   │   ├── test_text_languagemodelpreloader.py
│   │   ├── test_text_qrnn.py
│   │   ├── test_text_train.py
│   │   ├── test_text_transform.py
│   │   ├── test_torch_core.py
│   │   ├── test_train.py
│   │   ├── test_utils_fastai.py
│   │   ├── test_utils_links.py
│   │   ├── test_utils_mem.py
│   │   ├── test_utils_mod_independency.py
│   │   ├── test_utils.py
│   │   ├── test_vision_data.py
│   │   ├── test_vision_gan.py
│   │   ├── test_vision_image.py
│   │   ├── test_vision_learner.py
│   │   ├── test_vision_models_unet.py
│   │   ├── test_vision_train.py
│   │   ├── test_vision_transform.py
│   │   ├── test_widgets_image_cleaner.py
│   │   └── utils
│   │       ├── fakes.py
│   │       ├── mem.py
│   │       └── text.py
│   ├── tests_nb
│   │   ├── config.py
│   │   └── test_vision_train.ipynb
│   ├── tools
│   │   ├── build-docs
│   │   ├── checklink
│   │   │   ├── checklink-azure.yml
│   │   │   ├── checklink-course-v3.sh
│   │   │   ├── checklink-docs-local.sh
│   │   │   ├── checklink-docs.sh
│   │   │   ├── cookies.txt
│   │   │   ├── fastai-checklink
│   │   │   ├── fastai-checklink-run.sh
│   │   │   ├── README.md
│   │   │   └── run-checker.sh
│   │   ├── dep_graph.txt
│   │   ├── fastai-nbstripout
│   │   ├── gh-md-toc
│   │   ├── gh-md-toc-update-all
│   │   ├── make_sidebar.py
│   │   ├── README.md
│   │   ├── read-nbs
│   │   ├── run-after-git-clone
│   │   ├── trust-doc-nbs
│   │   ├── trust-doc-nbs-install-hook
│   │   └── trust-origin-git-config
│   └── tox.ini
├── github_example_images
│   ├── 100.png
│   ├── 134.png
│   ├── 138.png
│   ├── 144.png
│   ├── 148.png
│   ├── 14.png
│   ├── 159.png
│   ├── 167.png
│   ├── 170.png
│   ├── 175.png
│   ├── 184.png
│   ├── 191.png
│   ├── 196.png
│   ├── 197.png
│   ├── 22.png
│   ├── 254.png
│   ├── 257.png
│   ├── 263.png
│   ├── 264.png
│   ├── 266.png
│   ├── 27.png
│   ├── 48.png
│   ├── 56.png
│   ├── 61.png
│   ├── 83.png
│   ├── 87.png
│   ├── 90.png
│   └── 94.png
├── input_data
│   ├── trainA
│   └── trainB
├── README.md
└── utils
    └── rename_files.py

105 directories, 879 files
```

### Coming "soon"
* https://no.pinterest.com/pin/347973508707225607/ + landscapes
* cubism/polygon art
