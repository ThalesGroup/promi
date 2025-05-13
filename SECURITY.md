Describe here all the security policies in place on this repository to help your contributors to handle security issues efficiently.

## Goods practices to follow

:warning:**You must never store credentials information into source code or config file in a GitHub repository**
- Block sensitive data being pushed to GitHub by git-secrets or its likes as a git pre-commit hook
- Audit for slipped secrets with dedicated tools
- Use environment variables for secrets in CI/CD (e.g. GitHub Secrets) and secret managers in production

# Security Policy

## Supported Versions

| Version | Supported     |
| ------- | ------------- |
| `main`  | ✅ |
| others  | ❌ |

Only the `main` branch is supported for **academic/research use** under the Thales Non-Production License Agreement. No production or commercial usage is permitted.

## Reporting a Vulnerability

Use this section to tell people how to report a vulnerability.
Tell them where to go, how often they can expect to get an update on a reported vulnerability, what to expect if the vulnerability is accepted or declined, etc.

You can ask for support by contacting security@opensource.thalesgroup.com

## Disclosure policy

Define the procedure for what a reporter who finds a security issue needs to do in order to fully disclose the problem safely, including who to contact and how.

## Security Update policy

Define how you intend to update users about new security vulnerabilities as they are found.

## Security related configuration

Settings users should consider that would impact the security posture of deploying this project, such as HTTPS, authorization and many others.

## Known security gaps & future enhancements

Security improvements you haven’t gotten to yet.
Inform users those security controls aren’t in place, and perhaps suggest they contribute an implementation
