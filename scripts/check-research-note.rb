# Check the actual generated HTML and linked artifacts, without JavaScript.
require "nokogiri"
require "uri"

root = File.expand_path("..", __dir__)
site = File.join(root, "_site")
path = "/blog/2026/world-models/"
html = Nokogiri::HTML(File.read(File.join(site, path, "index.html")))
expected = {
  "citation_title" => "Feature Selection in World Models: A Spectral View",
  "citation_author" => "Hugues Van Assel",
  "citation_publication_date" => "2026/10/02",
  "citation_fulltext_html_url" => "https://huguesva.github.io#{path}",
  "citation_pdf_url" => "https://huguesva.github.io#{path}world-models.pdf"
}
expected.each do |name, content|
  values = html.css("head meta[name='#{name}']").map { |tag| tag["content"] }
  abort "Incorrect #{name}: #{values.inspect}" unless values == [content]
end
pdf_path = URI(expected.fetch("citation_pdf_url")).path
pdf = File.binread(File.join(site, pdf_path))
abort "Invalid PDF or PDF over 5 MB" unless pdf.start_with?("%PDF-") && pdf.bytesize < 5_000_000
abort "PDF must be alongside HTML" unless File.dirname(pdf_path) + "/" == path
bib_path = path + "citation.bib"
bib = File.read(File.join(site, bib_path)).strip
code = html.css("pre code").map { |node| node.text.strip }
abort "Downloadable BibTeX differs from the displayed citation" unless code.include?(bib)
[html, Nokogiri::HTML(File.read(File.join(site, "publications/index.html")))].each do |page|
  links = page.css("a[href]").map { |a| a["href"] }
  [pdf_path, bib_path].each { |link| abort "Missing link: #{link}" unless links.include?(link) }
end
puts "Research-note metadata, PDF, BibTeX, and discovery links passed."
