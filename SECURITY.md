# Security Policy

## Supported Versions

We actively support the following versions of ML101:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in ML101, please report it to us privately.

### How to Report

1. **Do not** create a public GitHub issue for security vulnerabilities
2. Email us at: ml101.security@gmail.com
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Any suggested fixes (if available)

### What to Expect

- **Initial Response**: We will acknowledge receipt of your report within 48 hours
- **Assessment**: We will assess the vulnerability and determine its severity
- **Fix**: We will work on a fix and coordinate the release
- **Credit**: We will credit you in our security advisory (unless you prefer to remain anonymous)

### Security Update Process

1. We will investigate and validate the reported vulnerability
2. We will develop and test a fix
3. We will release a security update
4. We will publish a security advisory
5. We will notify users through our communication channels

## Security Best Practices

When using ML101:

1. **Keep Updated**: Always use the latest version
2. **Validate Input**: Sanitize and validate all input data
3. **Secure Environment**: Use ML101 in a secure environment
4. **Monitor Dependencies**: Keep all dependencies updated
5. **Follow Principle of Least Privilege**: Run with minimal necessary permissions

## Known Security Considerations

- **Input Validation**: ML101 algorithms expect properly formatted numerical input
- **Memory Usage**: Large datasets may cause memory issues
- **Pickle Files**: Be cautious when loading saved models from untrusted sources
- **Dependencies**: Keep NumPy, SciPy, and other dependencies updated

## Contact

For security-related questions or concerns:
- Email: ml101.security@gmail.com
- For general questions: Create an issue on GitHub

Thank you for helping keep ML101 secure!
