<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

type Citation = {
  id: string
  label: string
  description: string
  required?: boolean
  reference: string
  doi: string
  bibtex: string
  metadataURL?: string
  zenodoRecord?: string
}

type PackageMetadata = {
  creators: Array<{ name: string }>
  title: string
}

type ZenodoRecord = {
  metadata: {
    publication_date: string
    version?: string
  }
}

type ZenodoRelease = {
  id: number
  links: { self_doi: string }
  metadata: PackageMetadata & {
    publication_date: string
    version?: string
  }
}

type ZenodoSearchResponse = {
  hits: { hits: ZenodoRelease[] }
}

const selected = ref<string[]>([])
const copyMessage = ref('')
const metadataError = ref('')
const justRelaxVersion = ref('')
const justRelaxVersions = ref<ZenodoRelease[]>([])

const citations = ref<Citation[]>([
  {
    id: 'justrelax-joss',
    label: 'JustRelax.jl JOSS article',
    description: 'Always cite this paper.',
    required: true,
    reference:
      'de Montserrat et al. (2026). JustRelax.jl: A Julia package for geodynamic modeling with matrix-free solvers. Journal of Open Source Software, 11(118), 9365.',
    doi: 'https://doi.org/10.21105/joss.09365',
    bibtex: `@article{deMontserrat2026,
  doi = {10.21105/joss.09365},
  url = {https://doi.org/10.21105/joss.09365},
  year = {2026},
  publisher = {The Open Journal},
  volume = {11},
  number = {118},
  pages = {9365},
  author = {de Montserrat, Albert and Aellig, Pascal S. and Schuler, Christian and Navarrete, Ivan and Räss, Ludovic and Fuchs, Lukas and Kaus, Boris J.p. and Dominguez, Hugo},
  title = {JustRelax.jl: A Julia package for geodynamic modeling with matrix-free solvers},
  journal = {Journal of Open Source Software}
}`,
  },
  {
    id: 'justrelax-zenodo',
    label: 'JustRelax.jl archived release',
    description: 'Always cite the archived version used for the work.',
    required: true,
    reference:
      'de Montserrat, A., Aellig, P. S., Schuler, C., Navarrete Jara, I., Räss, L., Fuchs, L., Kaus, B. J. P., & Dominguez, H. (2026). JustRelax.jl: A Julia package for geodynamic modeling with matrix-free solvers. Zenodo.',
    doi: 'https://doi.org/10.5281/zenodo.10212422',
    bibtex: `@misc{deMontserrat2026JustRelax,
  doi = {10.5281/zenodo.10212422},
  url = {https://doi.org/10.5281/zenodo.10212422},
  author = {de Montserrat, Albert and Aellig, Pascal S. and Schuler, Christian and Navarrete Jara, Iván and Räss, Ludovic and Fuchs, Lukas and Kaus, Boris J.P. and Dominguez, Hugo},
  title = {JustRelax.jl: A Julia package for geodynamic modeling with matrix-free solvers},
  publisher = {Zenodo},
  year = {2026}
}`,
  },
  {
    id: 'justpic',
    label: 'JustPIC.jl',
    description: 'Select if you used particle-in-cell advection.',
    reference: 'Loading the Zenodo citation…',
    doi: 'https://doi.org/10.5281/zenodo.10212675',
    bibtex: '',
    metadataURL: 'https://raw.githubusercontent.com/JuliaGeodynamics/JustPIC.jl/main/.zenodo.json',
    zenodoRecord: '10212675',
  },
  {
    id: 'geoparams',
    label: 'GeoParams.jl',
    description: 'Select if you used material properties or rheology models.',
    reference: 'Loading the Zenodo citation…',
    doi: 'https://doi.org/10.5281/zenodo.8089230',
    bibtex: '',
    metadataURL: 'https://raw.githubusercontent.com/JuliaGeodynamics/GeoParams.jl/main/.zenodo.json',
    zenodoRecord: '8089230',
  },
  {
    id: 'gmg',
    label: 'GeophysicalModelGenerator.jl',
    description: 'Select if you used it to prepare or visualize model data.',
    reference:
      'Kaus et al. (2024). GeophysicalModelGenerator.jl: A Julia package to visualise geoscientific data and create numerical model setups. Journal of Open Source Software, 9(103), 6763.',
    doi: 'https://doi.org/10.21105/joss.06763',
    bibtex: `@article{Kaus2024,
  doi = {10.21105/joss.06763},
  url = {https://doi.org/10.21105/joss.06763},
  year = {2024},
  publisher = {The Open Journal},
  volume = {9},
  number = {103},
  pages = {6763},
  author = {Kaus, Boris J.p. and Thielmann, Marcel and Aellig, Pascal and de Montserrat, Albert and de Siena, Luca and Frasukiewicz, Jacob and Fuchs, Lukas and Piccolo, Andrea and Ranocha, Hendrik and Riel, Nicolas and Schuler, Christian and Spang, Arne and Weiler, Tatjana},
  title = {GeophysicalModelGenerator.jl: A Julia package to visualise geoscientific data and create numerical model setups},
  journal = {Journal of Open Source Software}
}`,
  },
])

function escapeBibtex(value: string) {
  return value.replace(/[{}]/g, '\\$&')
}

function updateZenodoCitation(
  citation: Citation, packageMetadata: PackageMetadata, zenodoRecord: ZenodoRecord,
) {
  const authors = packageMetadata.creators.map(creator => creator.name)
  const leadAuthor = authors[0].split(',')[0]
  const shortAuthors = authors.length === 1 ? leadAuthor : `${leadAuthor} et al.`
  const year = zenodoRecord.metadata.publication_date.slice(0, 4)
  const version = zenodoRecord.metadata.version

  citation.reference = `${shortAuthors} (${year}). ${packageMetadata.title}${version ? ` (${version})` : ''}. Zenodo.`
  citation.bibtex = `@software{${citation.id}${year},
  doi = {${citation.doi.replace('https://doi.org/', '')}},
  url = {${citation.doi}},
  author = {${authors.join(' and ')}},
  title = {${escapeBibtex(packageMetadata.title)}},
  version = {${escapeBibtex(version ?? '')}},
  publisher = {Zenodo},
  year = {${year}}
}`
}

function updateJustRelaxCitation(release: ZenodoRelease) {
  const citation = citations.value.find(citation => citation.id === 'justrelax-zenodo')!
  const authors = release.metadata.creators.map(creator => creator.name)
  const leadAuthor = authors[0].split(',')[0]
  const shortAuthors = authors.length === 1 ? leadAuthor : `${leadAuthor} et al.`
  const year = release.metadata.publication_date.slice(0, 4)
  const version = release.metadata.version

  citation.doi = release.links.self_doi
  citation.description = 'Always cite the archived version used for the work.'
  citation.reference = `${shortAuthors} (${year}). ${release.metadata.title}${version ? ` (${version})` : ''}. Zenodo.`
  citation.bibtex = `@software{justrelax${year},
  doi = {${citation.doi.replace('https://doi.org/', '')}},
  url = {${citation.doi}},
  author = {${authors.join(' and ')}},
  title = {${escapeBibtex(release.metadata.title)}},
  version = {${escapeBibtex(version ?? '')}},
  publisher = {Zenodo},
  year = {${year}}
}`
}

function selectJustRelaxVersion() {
  const release = justRelaxVersions.value.find(
    release => String(release.id) === justRelaxVersion.value,
  )
  if (release) updateJustRelaxCitation(release)
}

async function fetchJustRelaxVersions() {
  const query = new URLSearchParams({
    q: 'metadata.title:"JustRelax.jl"',
    all_versions: 'true',
    size: '25',
  })
  const response = await fetch(`https://zenodo.org/api/records?${query}`)
  if (!response.ok) throw new Error('Unable to retrieve JustRelax release metadata.')

  const archiveVersions = ((await response.json()) as ZenodoSearchResponse).hits.hits
    .filter(release => release.metadata.title.startsWith('JustRelax.jl'))
    .sort((a, b) => b.metadata.publication_date.localeCompare(a.metadata.publication_date))
  const seenVersions = new Set<string>()
  justRelaxVersions.value = archiveVersions.filter(release => {
    const version = release.metadata.version?.replace(/^v/, '')
    if (!version || seenVersions.has(version)) return false
    seenVersions.add(version)
    return true
  })
  if (!justRelaxVersions.value.length) {
    throw new Error('No JustRelax releases were returned by Zenodo.')
  }
  justRelaxVersion.value = String(justRelaxVersions.value[0].id)
  selectJustRelaxVersion()
}

onMounted(async () => {
  try {
    await Promise.all(
      [
        fetchJustRelaxVersions(),
        ...citations.value
          .filter(citation => citation.metadataURL && citation.zenodoRecord)
          .map(async citation => {
            const [metadataResponse, recordResponse] = await Promise.all([
              fetch(citation.metadataURL!),
              fetch(`https://zenodo.org/api/records/${citation.zenodoRecord}`),
            ])
            if (!metadataResponse.ok || !recordResponse.ok) {
              throw new Error(`Unable to retrieve citation metadata for ${citation.label}`)
            }
            updateZenodoCitation(
              citation,
              (await metadataResponse.json()) as PackageMetadata,
              (await recordResponse.json()) as ZenodoRecord,
            )
          }),
      ],
    )
    selectJustRelaxVersion()
  } catch {
    metadataError.value = 'Unable to retrieve the current Zenodo citation metadata.'
  }
})

const selectedCitations = computed(() =>
  citations.value.filter(citation => citation.required || selected.value.includes(citation.id)),
)
const bibtex = computed(() =>
  selectedCitations.value
    .filter(citation => citation.bibtex)
    .map(citation => citation.bibtex)
    .join('\n\n'),
)

async function copyBibtex() {
  try {
    await navigator.clipboard.writeText(bibtex.value)
    copyMessage.value = 'BibTeX copied.'
  } catch {
    copyMessage.value = 'Select and copy the BibTeX block manually.'
  }
}
</script>

<template>
  <section class="citation-selector">
    <p>
      The JustRelax.jl JOSS paper and the archived release are included by
      default. Select every additional package that contributed to your work.
    </p>
    <p v-if="metadataError" class="citation-metadata-error" role="alert">
      {{ metadataError }}
    </p>

    <label class="citation-version-picker">
      <span><strong>JustRelax.jl version</strong></span>
      <select v-model="justRelaxVersion" :disabled="!justRelaxVersions.length" @change="selectJustRelaxVersion">
        <option v-if="!justRelaxVersions.length" value="">Loading releases…</option>
        <option v-for="release in justRelaxVersions" :key="release.id" :value="String(release.id)">
          {{ release.metadata.version }}
        </option>
      </select>
    </label>

    <fieldset>
      <legend>Packages used</legend>
      <label v-for="citation in citations" :key="citation.id" class="citation-option">
        <input
          v-if="!citation.required"
          v-model="selected"
          type="checkbox"
          :value="citation.id"
        >
        <input v-else type="checkbox" checked disabled>
        <span>
          <strong>{{ citation.label }}</strong>
          <span class="citation-option-description">{{ citation.description }}</span>
        </span>
      </label>
    </fieldset>

    <h2>References to cite</h2>
    <ol>
      <li v-for="citation in selectedCitations" :key="citation.id">
        {{ citation.reference }}
        <a :href="citation.doi">{{ citation.doi }}</a>
      </li>
    </ol>

    <div class="citation-bibtex-heading">
      <h2>BibTeX</h2>
      <button type="button" @click="copyBibtex">Copy BibTeX</button>
    </div>
    <p class="citation-copy-message" aria-live="polite">{{ copyMessage }}</p>
    <pre><code>{{ bibtex }}</code></pre>

    <p class="citation-version-note">
      The selector lists only Zenodo-backed JustRelax.jl releases. JustPIC.jl and
      GeoParams.jl use permanent Zenodo DOIs; their authors and title come from
      the packages' `.zenodo.json` files, and their year/version come from Zenodo.
    </p>
  </section>
</template>

<style scoped>
.citation-selector fieldset {
  margin: 1.5rem 0;
  padding: 1rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 0.5rem;
}

.citation-selector legend {
  padding: 0 0.25rem;
  font-weight: 600;
}

.citation-option {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
  margin: 0.75rem 0;
  cursor: pointer;
}

.citation-option input {
  margin-top: 0.3rem;
}

.citation-option-description {
  display: block;
  color: var(--vp-c-text-2);
}

.citation-version-picker {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  align-items: center;
  margin: 1.5rem 0;
}

.citation-version-picker select {
  padding: 0.4rem 0.6rem;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 0.25rem;
}

.citation-bibtex-heading {
  display: flex;
  gap: 1rem;
  align-items: center;
  justify-content: space-between;
}

.citation-bibtex-heading h2 {
  margin-bottom: 0;
}

.citation-copy-message {
  min-height: 1.5rem;
  margin: 0.25rem 0;
  color: var(--vp-c-text-2);
}

.citation-metadata-error {
  color: var(--vp-c-danger-1);
}

.citation-version-note {
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}
</style>
