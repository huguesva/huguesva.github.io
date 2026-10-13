# Emit citation metadata at build time; Scholar should not need JavaScript.
require "cgi"

Jekyll::Hooks.register :posts, :post_render do |post|
  next unless post.data["scholar"]

  absolute_url = ->(path) { "#{post.site.config.fetch('url')}#{post.site.baseurl}#{path}" }
  fields = [
    ["citation_title", post.data.fetch("title")],
    ["citation_publication_date", post.date.strftime("%Y/%m/%d")],
    ["citation_fulltext_html_url", absolute_url.call(post.url)]
  ]
  post.data.fetch("authors").each do |author|
    fields << ["citation_author", author.fetch("name")]
  end
  fields << ["citation_pdf_url", absolute_url.call(post.data.fetch("citation_pdf"))]
  tags = fields.map do |name, value|
    %(<meta name="#{name}" content="#{CGI.escapeHTML(value)}">)
  end.join("\n")
  post.output = post.output.sub("</head>", "#{tags}\n</head>")
end
