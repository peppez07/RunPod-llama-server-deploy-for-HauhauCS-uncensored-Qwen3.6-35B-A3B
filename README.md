# Experimental Ternary Bonsai / Prism Deployment — Use Is Discouraged

> **EXPERIMENTAL — DO NOT USE FOR PRODUCTION OR RELY ON THIS PROJECT.**
>
> This repository is an unfinished deployment experiment. Its author recommends
> avoiding its use. It has not been demonstrated to provide a reliable, secure,
> or suitable service. Do not deploy it for public access, integrate it into a
> product, or rely on its outputs for real decisions.

## Purpose and current status

This repository explores running
[OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF](https://huggingface.co/OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF)
using [Prism ML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp),
Docker, and RunPod.

The source and configuration are provided as an experimental record, not as a
recommendation to install, deploy, or operate them. The presence of a Dockerfile,
deployment script, example configuration, health route, or passing test does
not mean the project is ready for use.

**End-to-end Docker, GPU inference, and RunPod Serverless operation remain
unverified in this project's development session.** A previous RunPod source
build exceeded the platform's 30-minute build limit. The default Dockerfile was
subsequently changed to use a checksum-verified Prism binary release, but this
change has not established a successful GPU deployment.

Fifteen local Python tests passed when the deployment code was last checked.
Those tests use mocks for model downloads and infrastructure calls. They do not
prove model quality, hardware compatibility, security, production readiness, or
successful inference. Validation of the release archive's checksum and library
dependencies is also not a GPU execution test.

## Model and output disclosure

The selected model is described by its publisher as an uncensored/Heretic
variant with reduced safety alignment. This can increase the likelihood of
harmful, offensive, discriminatory, explicit, misleading, or otherwise
inappropriate output. The word “uncensored” is a model label, not a guarantee
about what it will answer or how it will behave.

Prism supplies runtime support for the model's ternary format. It does not
enable an “uncensored mode” or establish that the model is safe, accurate,
lawful to use in every setting, or appropriate for any audience.

The model may invent facts, sources, code, instructions, or confident
explanations. Generated code and commands may be insecure or destructive.
Outputs must not be treated as trusted instructions or authoritative advice.
No dependable moderation, factual verification, abuse prevention, or human
review system is provided by this repository.

Do not use it for medical, legal, financial, employment, education, security,
or other consequential decisions. Do not expose it to children or unsuspecting
users. Do not use it to make autonomous changes to systems or accounts.

## Security, privacy, and credentials

This project has not undergone an independent security audit. Authentication,
network isolation, credential handling, dependency integrity, and failure
recovery have not been comprehensively assessed.

Do not submit personal information, confidential material, proprietary data,
or credentials in prompts. Assume prompts, outputs, request metadata, and
configuration could be exposed through infrastructure, logs, misconfiguration,
or software defects. This repository makes no privacy or confidentiality
guarantee.

Hugging Face and RunPod tokens are credentials, not model settings to publish.
Never commit real tokens, bake them into Docker images, include them in example
files, or post them in issues or chat. Runtime secrets are still accessible to
the infrastructure that receives them. The example token fields are blank
intentionally.

The optional Serverless configuration relies on RunPod's gateway authentication.
That does not amount to a security assessment of the service. A health check
only reports a limited readiness condition; it does not establish safe or
correct model behavior.

## Infrastructure and cost disclosure

Deploying or testing this repository can create billable GPU, storage, image
build, and network usage. Failed builds, failed inference, startup delays, and
unhealthy workers may still involve costs. Stopping a client request or waiting
for a timeout does not necessarily stop the infrastructure or its charges.

The Pod launcher requests **150 GB of temporary container disk** by default.
RunPod Serverless storage must be configured separately in the endpoint console;
a Dockerfile or environment variable cannot allocate that disk. A repository
update does not automatically change an existing endpoint's storage settings.

Temporary files and model caches may disappear when workers or containers are
removed. Persistent volumes have different lifecycles and may continue to
incur charges. This project provides no assurance against data loss, duplicate
resources, unexpected downloads, capacity shortages, or unexpected charges.

If you have already deployed it, review your RunPod account, stop resources you
do not intend to keep running, and review retained storage and exposed endpoints.

## Dependencies, provenance, and limitations

The default experimental configuration references:

- Model file: `Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf`.
- Model revision: `5ab50fd49410941a8d4f97912303b1c7d458ffa2`.
- Prism binary release: `prism-b10826-c9c34bc`, Linux x86-64 / CUDA 12.8.
- Release archive SHA-256:
  `e4345c9fed6452701fd511f9837726f39a45bc95fe103b8cae55360562eadf7f`.

A checksum confirms agreement with the selected archive digest; it is not a
security audit or a guarantee of trustworthy behavior. Pinning a version does
not guarantee compatibility, continued availability, or freedom from defects.

The project depends on third-party model publishers, runtime maintainers,
container images, packages, GitHub, Hugging Face, NVIDIA, and RunPod. Their
software, services, access conditions, and availability can change. No
affiliation, sponsorship, approval, or endorsement by these parties is implied.

Third-party code and model weights remain subject to their own licenses,
acceptable-use requirements, and service terms. This notice does not grant
additional rights to those materials or replace their conditions.

## No warranty, support commitment, or suitability claim

This experimental material is supplied **as is**, without a promise of
correctness, availability, compatibility, security, maintenance, or fitness for
any purpose. No service-level commitment, deployment assistance commitment,
or guaranteed update schedule is offered.

These disclosures describe known limitations; they are not an exhaustive risk
assessment, a certification, or a substitute for independent evaluation.
The repository's recommendation remains: **avoid using or deploying this
project, and do not rely on its outputs.**
