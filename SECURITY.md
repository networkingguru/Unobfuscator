# Security Policy

## Reporting a Vulnerability

Please do **not** open a public GitHub issue for security vulnerabilities.

Report security issues privately using
[GitHub's private vulnerability reporting](https://github.com/networkingguru/Unobfuscator/security/advisories/new).
You can expect an acknowledgement within 72 hours.

---

## Scope

Unobfuscator is a local document processing tool. It:

- Makes outbound HTTP requests to **jmail.world**, **justice.gov**, and
  **archive.org** to fetch documents and metadata.
- Stores all state in a local SQLite database (`data/unobfuscator.db`).
- Does not expose any network service or API.

Security issues of interest include anything that could allow a malicious
document or API response to execute code or exfiltrate data from the host
machine.

---

## A Note on Subject Matter

The documents processed by Unobfuscator are public records released by the
U.S. Department of Justice. Unobfuscator does not redistribute any documents
or their contents. All processing happens locally on the user's machine.
