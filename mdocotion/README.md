# mdocotion

mdocotion is a mkdocs theme based on mkdocs-material with the looks and feels of Notion.

## How to integrate in Gitlab CI

1. Add this repository as a submodule to your mkdocs project.

```bash
git submodule add https://uva-hva.gitlab.host/hbo-ict/mdocotion.git mdocotion
```

2. Add the following lines to your `.gitlab-ci.yml` file:

```yaml
variables:
  GIT_SUBMODULE_STRATEGY: recursive
```

3. Configure the the as part of your mkdocs.yaml file:

```yaml
theme:
  name: material
  custom_dir: mdocotion/
```

## How to use locally

1. Clone this repository into your mkdocs project.

```bash
git clone https://uva-hva.gitlab.host/hbo-ict/mdocotion.git mdocotion
```

2. Install material for mkdocs.

```bash
pip install mkdocs-material
```

3. Add the following lines to your `mkdocs.yml` file.

```yaml
theme:
  name: material
  custom_dir: mdocotion/
```

4. Run mkdocs.

```bash
mkdocs serve
```

# Theme options

Here you can find options of the mkdocotion-theme.

## Full page header images

Add full-page header images to a page, start with installing the module:

(locally)
```bash
cd <your-mkdocs-project>/mdocotion
python setup.py install
```

(CI/CD)
```yaml
before_script:
  - time cd mdocotion && python setup.py install && cd ..
```


Following by adding mdocotion to the modules of the macros-plugin:

```yaml
plugins:
  - macros:
      modules: [mkdocs_macros_mdocotion]
```

And add this to the end of your markdown-file:

```markdown
{{ mdocotion_header('https://images.unsplash.com/photo-1565449446922-702749bcaf1b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1931&q=80') }}
```

## Colors navigation sidebar

You can change the color of the navigation sidebar by adding the following to your `mkdocs.yml` file:

```yaml
theme:
  name: material
  custom_dir: mdocotion/
  sidebar_color: 'rgb(100, 100, 100)'
```

where the value of sidebar_color is a valid value for the CSS property `background`.

## Colors header bar

You can change the color of the header bar by adding the following to your `mkdocs.yml` file:

```yaml
theme:
  name: material
  custom_dir: mdocotion/
  header_color: 'rgb(100, 100, 100)'
```

where the value of header_color is a valid value for the CSS property `background`.

## Top position content

You can change the position of the content by adding the following to your `mkdocs.yml` file:

```yaml
theme:
  name: material
  custom_dir: mdocotion/
  content_margin_top: 0px
```

where the value of content_margin_top is a valid value for the CSS property `margin-top`.

## Hide site title

You can hide the site title by adding the following to your `mkdocs.yml` file:

```yaml
theme:
  name: material
  custom_dir: mdocotion/
  hide_site_name: true
```