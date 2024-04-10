const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

app.post('/batch_generate', async (req, res) => {
    const { forward_url, prompt_token_id_list, n, top_p, top_k, temperature, max_tokens, stop, logprobs } = req.body;

    console.log("forward_url", forward_url);
    try {
        const promises = prompt_token_id_list.map(prompt_token_id =>
            axios.post(forward_url, {
                "prompt_token_ids": prompt_token_id,
                n,
                top_p,
                top_k,
                temperature,
                max_tokens,
                stop,
                logprobs,
            })
        );

        const responses = await Promise.all(promises);
        const combinedResults = responses.map(r => r.data);

        res.json(combinedResults);
    } catch (error) {
        res.status(500).send(error.toString());
    }
});

const PORT = process.env.PORT || 8003;
app.listen(PORT, () => console.log(`Server is running on port ${PORT}`));
