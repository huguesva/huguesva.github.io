# Hugues Van Assel — research website

Source for [huguesva.github.io](https://huguesva.github.io), a static academic website built with Jekyll and [al-folio](https://github.com/alshedivat/al-folio).

## Repository model

This repository deliberately contains only site-owned content and configuration. Theme layouts, JavaScript, fonts, and shared styles come from versioned al-folio Ruby plugins; they should not be copied back into the repository unless a small, intentional override is needed.

The main content lives in:

- `_pages/` — profile, publications, talks, blog index, and privacy notice
- `_posts/` — research articles
- `_bibliography/papers.bib` — publication metadata
- `_news/` — short announcements
- `assets/` — images and downloadable slides or posters
- `_config.yml` — identity, navigation, enabled features, and plugin configuration

Generated or machine-local directories such as `_site/`, `vendor/`, `node_modules/`, `.jekyll-cache/`, and virtual environments are ignored. Never commit them.

## Local development

Use Ruby 3.3.5 and Node.js 24. The version files work with common version managers.

```sh
bundle install
npm ci
bundle exec jekyll serve
```

Before pushing a change, run the same core checks as CI:

```sh
npm run format:check
bundle exec al-folio upgrade audit
JEKYLL_ENV=production bundle exec jekyll build --trace
npm run css:purge
```

Run `npm run format` to apply repository formatting.

## Deployment

Work in a short-lived branch and open a pull request into `master`. GitHub Actions builds and validates every pull request. A successful commit on `master` publishes the generated `_site` output to the `gh-pages` branch; generated files are never committed to `master`.

Dependabot checks Ruby gems, npm packages, and GitHub Actions weekly. Keep `Gemfile.lock` and `package-lock.json` committed so local and CI builds use reproducible dependencies.

## Updating al-folio

This site follows al-folio's v1 plugin architecture rather than maintaining a private copy of the theme. To adopt an upstream release:

1. Compare this repository's `Gemfile`, `_config.yml`, and workflow with the new al-folio starter.
2. Update the pinned `al_*` gem versions in `Gemfile` and run `bundle update`.
3. Run `bundle exec al-folio upgrade audit` and a production build.
4. Review the rendered profile, publications, talks, and both research posts before merging.

Prefer configuration and `assets/css/site.scss` over copying plugin-owned layouts. This keeps upstream updates small and reviewable.

## Licensing

The website code is available under the [MIT License](LICENSE). Unless noted otherwise, personal writing, publication material, images, slides, and posters remain copyright Hugues Van Assel and their respective co-authors.

## Citable research notes

The world-models article has build-time Google Scholar metadata, a downloadable
BibTeX entry, and a PDF alongside its canonical HTML URL. The `scholar: true`
front matter opts a post into `_plugins/scholar_metadata.rb`; `citation_pdf`
identifies its PDF. This does not guarantee Google Scholar indexing.

To regenerate the world-models PDF after editing the post or its included proofs:

```sh
JEKYLL_ENV=production bundle exec jekyll build
node scripts/export-world-models-pdf.mjs
JEKYLL_ENV=production bundle exec jekyll build
```

The export requires Node.js 24+ and Google Chrome on macOS, or Chromium on Linux.
Set `CHROME_BIN` to override the browser executable. It starts a temporary local
server and browser profile, moves all proofs and loss approximations into plain\nappendices with internal links, waits for equations and references,
and freezes animated figures at their first frame. The PDF links to the online
animations. Update the PDF version date in the export script when revising the note;
keep the original publication date consistent across the post and citation.
Commit the regenerated PDF with the content change. The second build copies that
PDF into `_site/`. No external repository deposit or Scholar profile entry is
created by this workflow.
